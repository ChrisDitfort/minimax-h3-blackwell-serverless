"""Tests for the PrivoraVideo product abstraction over MiniMax H3.

Run with:  python -m unittest discover -s tests -v

Two things are being pinned down.

The first is that the numbers this layer reports are the model's own. Every constant in
privora/canvas.py is a copy of one in ComfyUI's nodes_minimax_h3.py, and a copy of
somebody else's arithmetic rots silently - so the arithmetic is re-derived here against
the values read out of that node source at the pinned revision (dec5d945 / v0.30.2).

The second is that the abstraction holds. A caller sends prose and role metadata; H3 needs
<Picture 1> / <Video 1> / <Audio 2> tagging in a fixed presentation order. If the ordinals
drift out of step with the node's ordering, every reference instruction points at the
wrong file and the failure is invisible - the video just comes back subtly wrong.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privora import canvas, errors, prompt as prompt_module, references as refs  # noqa: E402
from privora import request as request_module  # noqa: E402


def reference_payload(kind, role, soundtrack=None):
    spec = {"type": kind, "role": role, "id": f"ref-{kind}-{role}"}
    if soundtrack:
        spec["soundtrack"] = {"type": "audio", "role": soundtrack}
    return spec


class CanvasArithmeticTests(unittest.TestCase):
    """The geometry must be the model's, not ours."""

    def test_the_frame_grid_is_17k_plus_5(self):
        for probe in range(5, 400):
            aligned = canvas.align_frame_count(probe)
            self.assertEqual(aligned % 17, 5)
            self.assertGreaterEqual(aligned, probe)
            self.assertLess(aligned - probe, 17, "snapping must go up to the next grid point")

    def test_an_already_legal_frame_count_is_unchanged(self):
        for legal in (5, 22, 39, 124, 362):
            self.assertEqual(canvas.align_frame_count(legal), legal)

    def test_the_native_canvas_matches_the_node(self):
        # adapt_canvas: 768 short edge, 768*1344 area cap, per-axis round to 32.
        self.assertEqual(canvas.adapt_canvas(16 / 9, 1.0), (1344, 768))
        self.assertEqual(canvas.adapt_canvas(9 / 16, 1.0), (768, 1344))
        self.assertEqual(canvas.adapt_canvas(1.0, 1.0), (768, 768))
        self.assertEqual(canvas.adapt_canvas(4 / 3, 1.0), (1024, 768))

    def test_every_canvas_is_legal(self):
        for ratio in canvas.ASPECT_RATIOS:
            for quality in canvas.QUALITY_TIERS:
                resolved = canvas.resolve_canvas(quality=quality, aspect_ratio=ratio)
                self.assertEqual(resolved.width % canvas.CANVAS_MULTIPLE, 0)
                self.assertEqual(resolved.height % canvas.CANVAS_MULTIPLE, 0)
                self.assertLessEqual(
                    resolved.width * resolved.height, canvas.MAX_PIXELS,
                    f"{quality} {ratio} exceeds the model's area cap",
                )
                self.assertEqual(resolved.frames % 17, 5)

    def test_the_measured_tiers_are_preserved(self):
        # Draft and Standard must reproduce the dimensions the existing benchmarks were
        # measured at, or the pre-rebuild baseline stops being comparable.
        draft = canvas.resolve_canvas(quality="draft", aspect_ratio="16:9")
        standard = canvas.resolve_canvas(quality="standard", aspect_ratio="16:9")
        self.assertEqual((draft.width, draft.height), (512, 288))
        self.assertEqual((standard.width, standard.height), (1024, 576))

    def test_the_reported_duration_is_the_real_one(self):
        resolved = canvas.resolve_canvas(duration_seconds=5)
        self.assertEqual(resolved.frames, 124)
        # 124/24 = 5.1667, not the 5 that was asked for. Reporting the request back would
        # make any caller labelling a clip wrong.
        self.assertAlmostEqual(resolved.duration_seconds, 124 / 24, places=4)
        self.assertTrue(resolved.duration_adjusted)
        self.assertEqual(resolved.as_metadata()["durationSeconds"], round(124 / 24, 4))

    def test_the_trained_range_is_flagged_not_refused(self):
        short = canvas.resolve_canvas(duration_seconds=2)
        self.assertTrue(short.outside_trained_range)
        self.assertIn("outsideTrainedRange", short.as_metadata())

        normal = canvas.resolve_canvas(duration_seconds=10)
        self.assertFalse(normal.outside_trained_range)
        self.assertNotIn("outsideTrainedRange", normal.as_metadata())

    def test_the_node_frame_ceiling_is_enforced(self):
        with self.assertRaises(ValueError):
            canvas.resolve_canvas(duration_seconds=200)  # 4800 frames > 3600

    def test_the_largest_legal_frame_count_is_derived_from_the_grid(self):
        maximum = canvas.maximum_aligned_frame_count()
        self.assertEqual(maximum, 3592)
        self.assertLessEqual(maximum, canvas.MAX_FRAMES)
        self.assertEqual(maximum % canvas.FRAME_GRID_MODULUS, canvas.FRAME_GRID_REMAINDER)
        self.assertGreater(canvas.align_frame_count(maximum + 1), canvas.MAX_FRAMES)

    def test_long_durations_are_available(self):
        # The node accepts up to 3600 frames and was trained to ~362 (~15s). The product
        # has only ever shipped ~5s, so this is real headroom rather than a limit.
        fifteen = canvas.resolve_canvas(duration_seconds=15)
        self.assertEqual(fifteen.frames, 362)
        self.assertFalse(fifteen.outside_trained_range)


class ReferenceLimitTests(unittest.TestCase):
    """Limits read from the node schema's autogrow templates, not from documentation."""

    def _set(self, images=0, videos=0, audios=0, soundtracks=0):
        collected = refs.ReferenceSet(
            images=[refs.Reference("image", "character") for _ in range(images)],
            videos=[refs.Reference("video", "motion") for _ in range(videos)],
            audios=[refs.Reference("audio", "voice") for _ in range(audios)],
        )
        for index in range(soundtracks):
            collected.videos[index].soundtrack = refs.Reference("audio", "ambience")
        return collected

    def test_nine_images_accepted_ten_rejected(self):
        refs.validate(self._set(images=9))
        with self.assertRaises(errors.PrivoraError) as caught:
            refs.validate(self._set(images=10))
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_COUNT)

    def test_three_videos_accepted_four_rejected(self):
        refs.validate(self._set(videos=3))
        with self.assertRaises(errors.PrivoraError):
            refs.validate(self._set(videos=4))

    def test_three_audio_accepted_four_rejected(self):
        refs.validate(self._set(audios=3))
        with self.assertRaises(errors.PrivoraError):
            refs.validate(self._set(audios=4))

    def test_the_total_cap_is_a_product_choice_not_a_model_limit(self):
        # 9 + 3 + 3 + 3 soundtracks = 18 inputs / 15 distinct files at the model level.
        # The product ships a lower cap, and the error says so, because a future policy
        # change must not look like a model change.
        self.assertEqual(refs.MODEL_MAX_REFERENCES, 18)
        self.assertGreater(refs.MODEL_MAX_REFERENCES, refs.PRODUCT_MAX_REFERENCES)

        refs.validate(self._set(images=9, videos=3))  # 12, at the cap
        with self.assertRaises(errors.PrivoraError) as caught:
            refs.validate(self._set(images=9, videos=3, audios=1))
        self.assertEqual(caught.exception.details["limit"], refs.PRODUCT_MAX_REFERENCES)
        self.assertEqual(caught.exception.details["modelLimit"], refs.MODEL_MAX_REFERENCES)

    def test_a_soundtrack_counts_toward_the_total(self):
        self.assertEqual(self._set(videos=3, soundtracks=3).file_count, 6)

    def test_an_unknown_role_is_rejected_with_the_allowed_set(self):
        collected = refs.ReferenceSet(images=[refs.Reference("image", "vibe")])
        with self.assertRaises(errors.PrivoraError) as caught:
            refs.validate(collected)
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_ROLE)
        self.assertIn("character", caught.exception.details["allowed"])

    def test_a_role_valid_for_another_type_is_still_rejected(self):
        collected = refs.ReferenceSet(images=[refs.Reference("image", "voice")])
        with self.assertRaises(errors.PrivoraError):
            refs.validate(collected)

    def test_video_duration_bounds(self):
        collected = refs.ReferenceSet(videos=[refs.Reference("video", "motion")])
        collected.videos[0].duration_seconds = 1.0
        with self.assertRaises(errors.PrivoraError) as caught:
            refs.validate_durations(collected)
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_DURATION)

        collected.videos[0].duration_seconds = 20.0
        with self.assertRaises(errors.PrivoraError):
            refs.validate_durations(collected)

        collected.videos[0].duration_seconds = 8.0
        refs.validate_durations(collected)

    def test_a_video_soundtrack_uses_the_audio_duration_limit(self):
        soundtrack = refs.Reference("audio", "ambience", duration_seconds=15.01)
        video = refs.Reference("video", "motion", duration_seconds=8.0,
                               soundtrack=soundtrack)
        with self.assertRaises(errors.PrivoraError) as caught:
            refs.validate_durations(refs.ReferenceSet(videos=[video]))
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_DURATION)
        self.assertEqual(caught.exception.details["type"], "soundtrack")

        soundtrack.duration_seconds = 15.0
        refs.validate_durations(refs.ReferenceSet(videos=[video]))

    def test_fidelity_maps_to_the_nodes_own_option(self):
        self.assertEqual(refs.resolve_fidelity("standard"), "match")
        self.assertEqual(refs.resolve_fidelity("high"), "max")
        self.assertEqual(refs.resolve_fidelity(None), "match")
        with self.assertRaises(errors.PrivoraError):
            refs.resolve_fidelity("ultra")


class SoundtrackContractTests(unittest.TestCase):
    def _parse(self, reference):
        return request_module.parse({
            "mode": "references", "prompt": "x", "references": [reference],
        })

    def test_the_canonical_nested_audio_shape_is_accepted(self):
        parsed = self._parse({
            "type": "video", "role": "motion", "id": "video-1",
            "soundtrack": {"type": "audio", "role": "ambience", "id": "audio-1"},
        })
        soundtrack = parsed.references.videos[0].soundtrack
        self.assertIsNotNone(soundtrack)
        self.assertEqual((soundtrack.type, soundtrack.role), ("audio", "ambience"))
        self.assertEqual(parsed.references.file_count, 2)

    def test_soundtrack_is_rejected_on_image_and_audio_references(self):
        for parent_type, parent_role in (("image", "character"), ("audio", "music")):
            with self.subTest(parent_type=parent_type):
                with self.assertRaises(errors.PrivoraError) as caught:
                    self._parse({
                        "type": parent_type, "role": parent_role,
                        "soundtrack": {"type": "audio", "role": "ambience"},
                    })
                self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)

    def test_soundtrack_must_explicitly_be_audio(self):
        for soundtrack in (
            {"role": "ambience"},
            {"type": "image", "role": "general"},
            {"type": "video", "role": "general"},
        ):
            with self.subTest(soundtrack=soundtrack):
                with self.assertRaises(errors.PrivoraError) as caught:
                    self._parse({"type": "video", "role": "motion",
                                 "soundtrack": soundtrack})
                self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)

    def test_soundtrack_must_be_an_object_with_an_audio_role(self):
        for soundtrack in (
            ["not", "an", "object"],
            {"type": "audio", "role": "character"},
        ):
            with self.subTest(soundtrack=soundtrack):
                with self.assertRaises(errors.PrivoraError) as caught:
                    self._parse({"type": "video", "role": "motion",
                                 "soundtrack": soundtrack})
                self.assertIn(caught.exception.code,
                              (errors.INVALID_REFERENCE_TYPE, errors.INVALID_REFERENCE_ROLE))

    def test_reference_validator_defends_the_soundtrack_shape_too(self):
        for parent in (
            refs.Reference("image", "character", soundtrack=refs.Reference("audio")),
            refs.Reference("audio", "music", soundtrack=refs.Reference("audio")),
        ):
            with self.subTest(parent=parent.type):
                collected = refs.ReferenceSet(
                    images=[parent] if parent.type == "image" else [],
                    audios=[parent] if parent.type == "audio" else [],
                )
                with self.assertRaises(errors.PrivoraError):
                    refs.validate(collected)


class OrdinalTests(unittest.TestCase):
    """Ordinals must match the node's presentation order exactly."""

    def test_a_video_soundtrack_takes_the_first_audio_ordinal(self):
        # The node emits each soundtrack's <Audio j> immediately before its <Video k>, and
        # standalone audio continues that counter. Numbering standalone audio from 1 would
        # point every audio instruction at the wrong clip.
        collected = refs.ReferenceSet(
            videos=[refs.Reference("video", "motion")],
            audios=[refs.Reference("audio", "voice")],
        )
        collected.videos[0].soundtrack = refs.Reference("audio", "ambience")
        refs.assign_ordinals(collected)

        self.assertEqual(collected.videos[0].soundtrack.tag, "Audio 1")
        self.assertEqual(collected.audios[0].tag, "Audio 2")
        self.assertEqual(collected.videos[0].tag, "Video 1")

    def test_ordering_is_images_then_videos_then_audio(self):
        collected = refs.ReferenceSet(
            images=[refs.Reference("image", "character"), refs.Reference("image", "clothing")],
            videos=[refs.Reference("video", "motion")],
            audios=[refs.Reference("audio", "music")],
        )
        refs.assign_ordinals(collected)
        self.assertEqual([r.tag for r in collected.all],
                         ["Picture 1", "Picture 2", "Video 1", "Audio 1"])


class PromptCompilerTests(unittest.TestCase):
    def test_the_compiled_prompt_is_deterministic(self):
        payload = {
            "mode": "references", "prompt": "The woman walks through the room", "seed": 7,
            "references": [reference_payload("image", "character"),
                           reference_payload("video", "motion", soundtrack="ambience")],
            "camera": {"shot": "medium", "movement": "orbit"},
        }
        first = request_module.parse(dict(payload)).prompt.text
        second = request_module.parse(dict(payload)).prompt.text
        self.assertEqual(first, second)

    def test_roles_become_tagged_instructions(self):
        compiled = request_module.parse({
            "mode": "references", "prompt": "A scene", "seed": 1,
            "references": [reference_payload("image", "character"),
                           reference_payload("image", "clothing"),
                           reference_payload("audio", "voice")],
        }).prompt
        self.assertIn("Use the character from <Picture 1>.", compiled.text)
        self.assertIn("Use the clothing from <Picture 2>.", compiled.text)
        self.assertIn("Use the voice from <Audio 1>.", compiled.text)

    def test_user_text_cannot_impersonate_a_reference_tag(self):
        # A caller writing "<Picture 3>" with two images supplied would otherwise point
        # the model at a reference that does not exist.
        compiled = request_module.parse({
            "mode": "references", "prompt": "Match the jacket in <Picture 3> exactly", "seed": 1,
            "references": [reference_payload("image", "clothing")],
        }).prompt
        self.assertTrue(compiled.neutralised_user_tags)
        self.assertNotIn("<Picture 3>", compiled.text)
        self.assertIn("picture 3", compiled.text)
        self.assertIn("<Picture 1>", compiled.text, "our own tag must survive")

    def test_a_general_role_adds_no_instruction(self):
        compiled = request_module.parse({
            "mode": "references", "prompt": "A scene", "seed": 1,
            "references": [reference_payload("image", "general")],
        }).prompt
        self.assertEqual(compiled.text, "A scene.")
        self.assertEqual(compiled.tag_roles, {"<Picture 1>": "general"})

    def test_camera_and_style_compile_to_prose(self):
        compiled = prompt_module.compile_prompt(
            "A street", None,
            camera={"shot": "wide", "movement": "dolly", "speed": "slow"},
            style={"visual": "noir", "lighting": "neon"},
        )
        self.assertIn("Filmed as a wide shot", compiled.text)
        self.assertIn("Style: film noir, neon lighting.", compiled.text)

    def test_unknown_camera_values_are_dropped_not_invented(self):
        self.assertEqual(prompt_module.compile_camera({"shot": "bird", "movement": "vibes"}), "")


class ModeRoutingTests(unittest.TestCase):
    def test_create_routes_to_fl2va(self):
        self.assertEqual(request_module.parse({"mode": "create", "prompt": "x"}).family, "fl2va")

    def test_animate_routes_to_fl2va_with_either_frame(self):
        first = request_module.parse(
            {"mode": "animate", "prompt": "x", "firstFrame": {"type": "image", "id": "a"}}
        )
        self.assertEqual(first.family, "fl2va")
        self.assertIsNotNone(first.first_frame)
        self.assertIsNone(first.last_frame)

        last = request_module.parse(
            {"mode": "animate", "prompt": "x", "lastFrame": {"type": "image", "id": "b"}}
        )
        self.assertIsNone(last.first_frame)
        self.assertIsNotNone(last.last_frame)

        both = request_module.parse({
            "mode": "animate", "prompt": "x",
            "firstFrame": {"type": "image", "id": "a"}, "lastFrame": {"type": "image", "id": "b"},
        })
        self.assertEqual(both.as_metadata()["keyframes"], {"first": True, "last": True})

    def test_animate_needs_at_least_one_frame(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({"mode": "animate", "prompt": "x"})
        self.assertEqual(caught.exception.code, errors.MISSING_FRAME)

    def test_references_and_remix_route_to_ref2va(self):
        for mode in ("references", "remix"):
            parsed = request_module.parse({
                "mode": mode, "prompt": "x",
                "references": [reference_payload("video", "source")],
            })
            self.assertEqual(parsed.family, "ref2va", mode)

    def test_references_need_at_least_one_reference(self):
        with self.assertRaises(errors.PrivoraError):
            request_module.parse({"mode": "references", "prompt": "x"})

    def test_create_rejects_keyframes(self):
        for field in ("firstFrame", "lastFrame"):
            with self.subTest(field=field):
                with self.assertRaises(errors.PrivoraError) as caught:
                    request_module.parse({
                        "mode": "create", "prompt": "x",
                        field: {"type": "image", "id": "frame"},
                    })
                self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)

    def test_animate_rejects_references_even_with_a_keyframe(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({
                "mode": "animate", "prompt": "x",
                "firstFrame": {"type": "image", "id": "frame"},
                "references": [reference_payload("image", "character")],
            })
        self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)

    def test_references_rejects_keyframes(self):
        for field in ("firstFrame", "lastFrame"):
            with self.subTest(field=field):
                with self.assertRaises(errors.PrivoraError) as caught:
                    request_module.parse({
                        "mode": "references", "prompt": "x",
                        "references": [reference_payload("image", "character")],
                        field: {"type": "image", "id": "frame"},
                    })
                self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)

    def test_remix_requires_a_source_role_video(self):
        invalid_sets = (
            [reference_payload("image", "character")],
            [reference_payload("audio", "music")],
            [reference_payload("video", "motion")],
            [reference_payload("video", "motion"), reference_payload("image", "style")],
        )
        for references in invalid_sets:
            with self.subTest(references=references):
                with self.assertRaises(errors.PrivoraError) as caught:
                    request_module.parse({"mode": "remix", "prompt": "x",
                                          "references": references})
                self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_COUNT)
                self.assertEqual(caught.exception.details["required"],
                                 {"type": "video", "role": "source"})

    def test_remix_accepts_a_source_video_and_rejects_keyframes(self):
        payload = {
            "mode": "remix", "prompt": "x",
            "references": [reference_payload("video", "source")],
        }
        self.assertEqual(request_module.parse(payload).family, "ref2va")
        with self.assertRaises(errors.PrivoraError):
            request_module.parse({**payload, "firstFrame": {"type": "image", "id": "f"}})

    def test_keyframes_must_be_images(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({
                "mode": "animate", "prompt": "x",
                "firstFrame": {"type": "video", "id": "not-an-image"},
            })
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)

    def test_create_rejects_references_rather_than_ignoring_them(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({
                "mode": "create", "prompt": "x",
                "references": [reference_payload("image", "character")],
            })
        self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)

    def test_an_unknown_mode_lists_the_supported_ones(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({"mode": "upscale", "prompt": "x"})
        self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)
        self.assertIn("create", caught.exception.details["supported"])

    def test_remix_is_documented_as_regeneration_not_editing(self):
        note = request_module.MODE_NOTES["remix"]
        self.assertIn("not deterministic video editing", note)


class SeedTests(unittest.TestCase):
    def test_an_explicit_seed_is_returned_unchanged(self):
        self.assertEqual(request_module.parse({"mode": "create", "prompt": "x", "seed": 51}).seed, 51)

    def test_an_omitted_seed_is_generated_and_reported(self):
        parsed = request_module.parse({"mode": "create", "prompt": "x"})
        self.assertIsInstance(parsed.seed, int)
        self.assertEqual(parsed.as_metadata()["seed"], parsed.seed)

    def test_the_seed_ceiling_matches_the_node(self):
        request_module.parse({"mode": "create", "prompt": "x", "seed": request_module.MAX_SEED})
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({"mode": "create", "prompt": "x",
                                  "seed": request_module.MAX_SEED + 1})
        self.assertEqual(caught.exception.code, errors.INVALID_SEED)


class LegacyCompatibilityTests(unittest.TestCase):
    """The pre-rebuild schema must keep working, or rollback comparison dies with it."""

    LEGACY = {
        "prompt": "A cinematic ocean scene",
        "width": 1024, "height": 576, "frames": 124, "steps": 20, "seed": 51,
    }

    def test_a_legacy_request_is_accepted(self):
        parsed = request_module.parse(dict(self.LEGACY))
        self.assertTrue(parsed.legacy)
        self.assertEqual(parsed.mode, "create")
        self.assertEqual((parsed.canvas.width, parsed.canvas.height), (1024, 576))
        self.assertEqual(parsed.canvas.frames, 124)
        self.assertEqual(parsed.canvas.steps, 20)
        self.assertEqual(parsed.seed, 51)

    def test_legacy_dimensions_are_used_verbatim(self):
        # 1280x704 is the measured HD baseline and is not exactly 16:9. It must run as
        # asked rather than being re-derived through the tier table.
        parsed = request_module.parse(dict(self.LEGACY, width=1280, height=704))
        self.assertEqual((parsed.canvas.width, parsed.canvas.height), (1280, 704))

    def test_a_legacy_image_input_becomes_a_first_frame(self):
        parsed = request_module.parse(dict(self.LEGACY, image_url="https://example.com/a.png"))
        self.assertEqual(parsed.mode, "animate")
        self.assertIsNotNone(parsed.first_frame)

    def test_legacy_metadata_reports_the_new_envelope(self):
        metadata = request_module.parse(dict(self.LEGACY)).as_metadata()
        for field in ("width", "height", "frames", "fps", "durationSeconds", "seed", "mode"):
            self.assertIn(field, metadata)
        self.assertEqual(metadata["schema"], "legacy")

    def test_legacy_steps_are_limited_to_the_existing_one_to_one_hundred_range(self):
        for supplied in (0, 101, True, "not-a-number"):
            with self.subTest(supplied=supplied):
                with self.assertRaises(errors.PrivoraError) as caught:
                    request_module.parse(dict(self.LEGACY, steps=supplied))
                self.assertEqual(caught.exception.code, errors.INVALID_STEPS)

    def test_legacy_parser_remains_permissive_about_unrelated_fields(self):
        parsed = request_module.parse(dict(self.LEGACY, backend="runpod", fast=True))
        self.assertTrue(parsed.legacy)


class CanonicalFieldTests(unittest.TestCase):
    def test_the_stable_top_level_allowlist_is_exact(self):
        self.assertEqual(request_module.CANONICAL_FIELDS, {
            "mode", "prompt", "quality", "aspectRatio", "duration", "seed",
            "generationMode", "firstFrame", "lastFrame", "references",
            "referenceFidelity", "camera", "style", "privacy", "encryption",
            "progress", "output",
        })

    def test_a_likely_misspelling_is_rejected_instead_of_falling_back(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({
                "mode": "create", "prompt": "x", "generatonMode": "turbo",
            })
        self.assertEqual(caught.exception.code, errors.UNKNOWN_FIELD)
        self.assertEqual(caught.exception.details["fields"], ["generatonMode"])

    def test_unknown_field_values_and_hostile_names_are_not_echoed(self):
        secret = "REFERENCE_TOKEN_CANARY_4E2A"
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({"mode": "create", "prompt": "x", secret: secret})
        rendered = str(caught.exception.as_response())
        self.assertNotIn(secret, rendered)


class ErrorSanitisationTests(unittest.TestCase):
    """A rejection must never carry the prompt or a filename back to the caller."""

    CANARY = "PROMPT_CANARY_PRIVORA_7F91A2"

    def test_a_rejected_request_does_not_echo_the_prompt(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({
                "mode": "references", "prompt": f"{self.CANARY} a scene",
                "references": [reference_payload("image", "character") for _ in range(10)],
            })
        rendered = caught.exception.as_response()
        self.assertNotIn(self.CANARY, str(rendered))
        self.assertNotIn(self.CANARY, caught.exception.as_log_line())

    def test_the_response_carries_a_stable_code(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({"mode": "create", "prompt": "x", "quality": "8k"})
        response = caught.exception.as_response()
        self.assertEqual(response["errorCode"], errors.INVALID_QUALITY)
        self.assertTrue(caught.exception.is_client_error)

    def test_internal_detail_is_logged_but_never_returned(self):
        error = errors.PrivoraError(
            errors.GENERATION_FAILED, "Generation failed.",
            internal=f"node blew up on {self.CANARY}",
        )
        self.assertNotIn(self.CANARY, str(error.as_response()))
        self.assertIn(self.CANARY, error.as_log_line())


class CapabilitiesTests(unittest.TestCase):
    def test_capabilities_reports_ref2va_availability_honestly(self):
        without = request_module.capabilities(ref2va_available=False)
        self.assertFalse(without["modes"]["references"]["available"])
        self.assertTrue(without["modes"]["create"]["available"])

        with_ref = request_module.capabilities(ref2va_available=True)
        self.assertTrue(with_ref["modes"]["references"]["available"])

    def test_capabilities_states_both_reference_ceilings(self):
        reported = request_module.capabilities(ref2va_available=True)["references"]
        self.assertEqual(reported["maxTotal"], refs.PRODUCT_MAX_REFERENCES)
        self.assertEqual(reported["modelMaxTotal"], refs.MODEL_MAX_REFERENCES)

    def test_capabilities_lists_every_ratio_and_tier(self):
        reported = request_module.capabilities(ref2va_available=True)
        self.assertEqual(set(reported["aspectRatios"]), set(canvas.ASPECT_RATIOS))
        self.assertEqual(set(reported["quality"]), set(canvas.QUALITY_TIERS))

    def test_advertised_max_duration_is_executable_and_the_next_grid_is_not(self):
        maximum = request_module.capabilities(ref2va_available=True)["duration"]["maxSeconds"]
        parsed = request_module.parse({"mode": "create", "prompt": "x", "duration": maximum})
        self.assertEqual(parsed.canvas.frames, canvas.maximum_aligned_frame_count())
        self.assertLessEqual(parsed.canvas.frames, canvas.MAX_FRAMES)

        with self.assertRaises(errors.PrivoraError) as caught:
            request_module.parse({"mode": "create", "prompt": "x", "duration": 150})
        self.assertEqual(caught.exception.code, errors.INVALID_DURATION)


if __name__ == "__main__":
    unittest.main()
