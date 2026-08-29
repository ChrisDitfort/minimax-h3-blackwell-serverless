"""Model routing, Turbo selection and the ComfyUI graphs that come out the other end.

Run with:  python -m unittest discover -s tests -v

The thing worth protecting here is that a *product* choice - "turbo" - lands on the right
checkpoint, the right LoRA and the right step count, and that an unavailable combination is
refused before 21 GB of weights get loaded rather than after.

The Ref2VA asymmetry is the interesting case and has its own tests: FL2VA has 8-step and
4-step Turbo LoRAs, Ref2VA has only 4-step. `turbo` therefore does not exist for reference
generation, and the worker must say so rather than quietly substituting something else.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privora import canvas, errors, models, workflows  # noqa: E402
from privora import request as request_module  # noqa: E402

FL2VA_CKPT = models.CHECKPOINTS[models.FL2VA]
REF2VA_CKPT = models.CHECKPOINTS[models.REF2VA]
LORA_8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
LORA_4 = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
LORA_REF4 = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

FULL = models.ModelInventory.from_names([FL2VA_CKPT, REF2VA_CKPT, LORA_8, LORA_4, LORA_REF4])
FL2VA_ONLY = models.ModelInventory.from_names([FL2VA_CKPT])
NO_LORAS = models.ModelInventory.from_names([FL2VA_CKPT, REF2VA_CKPT])


def create(**overrides):
    payload = {"mode": "create", "prompt": "An ocean scene", "seed": 51}
    payload.update(overrides)
    return request_module.parse(payload)


def references(**overrides):
    payload = {
        "mode": "references", "prompt": "A woman walks", "seed": 51,
        "references": [{"type": "image", "role": "character"}],
    }
    payload.update(overrides)
    return request_module.parse(payload)


class GenerationModeParsingTests(unittest.TestCase):
    def test_the_wire_spelling_is_camel_case(self):
        self.assertEqual(models.parse_generation_mode("turboFast"), models.TURBO_FAST)
        self.assertEqual(models.parse_generation_mode("turbo_fast"), models.TURBO_FAST)
        self.assertEqual(models.parse_generation_mode("quality"), models.QUALITY)

    def test_the_default_is_quality(self):
        self.assertEqual(models.parse_generation_mode(None), models.QUALITY)
        self.assertEqual(create().generation_mode, models.QUALITY)

    def test_an_arbitrary_step_count_is_not_a_generation_mode(self):
        # The frontend must never send raw step counts; 1-100 is not a product concept.
        with self.assertRaises(errors.PrivoraError) as caught:
            models.parse_generation_mode("8")
        self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)


class TurboRoutingTests(unittest.TestCase):
    def test_fl2va_quality_uses_base_weights_and_twenty_steps(self):
        plan = workflows.build(create(), FULL, models.QUALITY)
        self.assertEqual(plan.graph["unet"]["inputs"]["unet_name"], FL2VA_CKPT)
        self.assertNotIn("lora", plan.graph)
        self.assertEqual(plan.graph["sigmas"]["inputs"]["steps"], 20)
        self.assertEqual(plan.as_metadata()["acceleration"], "none")

    def test_fl2va_turbo_selects_the_eight_step_lora(self):
        plan = workflows.build(create(), FULL, models.TURBO)
        self.assertEqual(plan.graph["lora"]["inputs"]["lora_name"], LORA_8)
        self.assertEqual(plan.graph["sigmas"]["inputs"]["steps"], 8)
        self.assertEqual(plan.as_metadata(),
                         {"generationMode": "turbo", "steps": 8, "acceleration": "turbo_lora"})

    def test_fl2va_turbo_fast_selects_the_four_step_lora(self):
        plan = workflows.build(create(), FULL, models.TURBO_FAST)
        self.assertEqual(plan.graph["lora"]["inputs"]["lora_name"], LORA_4)
        self.assertEqual(plan.graph["sigmas"]["inputs"]["steps"], 4)
        self.assertIn("768p", plan.as_metadata()["accelerationNote"])

    def test_the_lora_is_applied_to_the_model_only(self):
        # These are distilled model LoRAs with no text-encoder half. Routing CLIP through
        # a loader that expects one would be wrong.
        plan = workflows.build(create(), FULL, models.TURBO)
        self.assertEqual(plan.graph["lora"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(plan.graph["lora"]["inputs"]["model"], ["unet", 0])
        self.assertEqual(plan.graph["guider"]["inputs"]["model"], ["lora", 0])
        self.assertEqual(plan.graph["sigmas"]["inputs"]["model"], ["lora", 0])

    def test_without_a_lora_the_guider_reads_the_unet_directly(self):
        plan = workflows.build(create(), FULL, models.QUALITY)
        self.assertEqual(plan.graph["guider"]["inputs"]["model"], ["unet", 0])


class Ref2vaTurboTests(unittest.TestCase):
    """Ref2VA has a 4-step LoRA and no 8-step one. Verified, not assumed."""

    def test_references_select_the_ref2va_checkpoint(self):
        plan = workflows.build(references(), FULL, models.QUALITY)
        self.assertEqual(plan.graph["unet"]["inputs"]["unet_name"], REF2VA_CKPT)
        self.assertEqual(plan.graph["conditioning"]["class_type"], "MiniMaxH3ReferenceToVideo")

    def test_ref2va_turbo_fast_uses_the_ref2va_lora(self):
        plan = workflows.build(references(), FULL, models.TURBO_FAST)
        self.assertEqual(plan.graph["lora"]["inputs"]["lora_name"], LORA_REF4)
        self.assertEqual(plan.graph["sigmas"]["inputs"]["steps"], 4)
        self.assertIn("v0.1", plan.as_metadata()["accelerationNote"])

    def test_ref2va_turbo_is_refused_because_no_eight_step_lora_exists(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            workflows.build(references(), FULL, models.TURBO)
        self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)
        self.assertEqual(caught.exception.details["available"], ["quality", "turboFast"])

    def test_no_eight_step_ref2va_entry_exists_at_all(self):
        self.assertNotIn((models.REF2VA, models.TURBO), models.ACCELERATIONS)


class AvailabilityTests(unittest.TestCase):
    """Capabilities describe the image that was built, not the one we meant to build."""

    def test_a_missing_checkpoint_makes_its_modes_unavailable(self):
        self.assertFalse(FL2VA_ONLY.has_family(models.REF2VA))
        self.assertEqual(FL2VA_ONLY.available_modes(models.REF2VA), [])
        with self.assertRaises(errors.PrivoraError) as caught:
            workflows.build(references(), FL2VA_ONLY, models.QUALITY)
        self.assertEqual(caught.exception.code, errors.MODEL_LOAD_FAILED)

    def test_missing_loras_leave_only_quality(self):
        self.assertEqual(NO_LORAS.available_modes(models.FL2VA), ["quality"])
        with self.assertRaises(errors.PrivoraError) as caught:
            workflows.build(create(), NO_LORAS, models.TURBO)
        self.assertEqual(caught.exception.code, errors.MODEL_LOAD_FAILED)

    def test_the_capability_report_matches_the_built_image(self):
        described = FULL.describe()
        self.assertEqual(described["models"], {"fl2va": True, "ref2va": True})
        self.assertEqual(described["turbo"]["fl2va"], {"8step": True, "4step": True})
        # No 8step key for ref2va at all - reporting False would imply it exists but is
        # missing from this build, which is a different and untrue claim.
        self.assertEqual(described["turbo"]["ref2va"], {"4step": True})
        self.assertEqual(described["byFamily"]["ref2va"], ["quality", "turboFast"])

    def test_an_fl2va_only_image_reports_honestly(self):
        described = FL2VA_ONLY.describe()
        self.assertFalse(described["models"]["ref2va"])
        self.assertFalse(described["generationModes"]["turbo"])

    def test_rejection_happens_before_any_graph_is_built(self):
        # The point of resolving first: an unavailable combination costs milliseconds,
        # not a 21 GB checkpoint load.
        with self.assertRaises(errors.PrivoraError):
            workflows.build(references(), FL2VA_ONLY, models.TURBO_FAST)


class LegacyStepsTests(unittest.TestCase):
    def test_legacy_twenty_steps_maps_to_the_base_workflow(self):
        parsed = request_module.parse(
            {"prompt": "x", "width": 1024, "height": 576, "frames": 124, "steps": 20, "seed": 1}
        )
        self.assertTrue(parsed.legacy)
        self.assertEqual(parsed.generation_mode, models.QUALITY)
        plan = workflows.build(parsed, FULL, parsed.generation_mode)
        self.assertNotIn("lora", plan.graph)
        self.assertEqual(plan.graph["sigmas"]["inputs"]["steps"], 20)

    def test_every_legal_explicit_step_count_reaches_the_scheduler_verbatim(self):
        # Legacy steps select base-model sampling depth, never a distilled Turbo LoRA.
        for steps in (4, 8, 14, 20, 30):
            with self.subTest(steps=steps):
                parsed = request_module.parse({
                    "prompt": "x", "width": 1024, "height": 576,
                    "frames": 124, "steps": steps, "seed": 1,
                })
                self.assertEqual(parsed.generation_mode, models.QUALITY)
                plan = workflows.build(parsed, FULL, parsed.generation_mode)
                self.assertNotIn("lora", plan.graph)
                self.assertEqual(plan.graph[workflows.SIGMAS]["inputs"]["steps"], steps)
                self.assertEqual(plan.as_metadata(), {
                    "generationMode": "quality", "steps": steps, "acceleration": "none",
                })

    def test_steps_to_generation_mode_only_recognises_the_base_count(self):
        self.assertEqual(models.steps_to_generation_mode(models.FL2VA, 20), models.QUALITY)
        self.assertIsNone(models.steps_to_generation_mode(models.FL2VA, 8))
        self.assertIsNone(models.steps_to_generation_mode(models.FL2VA, 4))


class CapabilityExecutionConsistencyTests(unittest.TestCase):
    """Every advertised product choice must parse and build on the described inventory."""

    def setUp(self):
        self.capabilities = request_module.capabilities(ref2va_available=True)
        self.capabilities.update(FULL.describe())

    def _request_for_mode(self, mode, **extra):
        payload = {"mode": mode, "prompt": "x", "seed": 1, **extra}
        if mode == "animate":
            payload["firstFrame"] = {"type": "image", "id": "frame"}
        elif mode == "references":
            payload["references"] = [{"type": "image", "role": "character", "id": "ref"}]
        elif mode == "remix":
            payload["references"] = [{"type": "video", "role": "source", "id": "source"}]
        return request_module.parse(payload)

    def test_every_advertised_mode_parses_and_has_a_builder(self):
        for mode, described in self.capabilities["modes"].items():
            with self.subTest(mode=mode):
                self.assertTrue(described["available"])
                request = self._request_for_mode(mode)
                plan = workflows.build(request, FULL, request.generation_mode)
                expected = ("MiniMaxH3ImageToVideo" if request.family == models.FL2VA
                            else "MiniMaxH3ReferenceToVideo")
                self.assertEqual(plan.graph[workflows.CONDITIONING]["class_type"], expected)

    def test_every_advertised_tier_and_ratio_resolves_to_the_reported_canvas(self):
        for quality, tier in self.capabilities["quality"].items():
            for ratio, dimensions in tier["dimensions"].items():
                with self.subTest(quality=quality, ratio=ratio):
                    request = self._request_for_mode(
                        "create", quality=quality, aspectRatio=ratio
                    )
                    plan = workflows.build(request, FULL, request.generation_mode)
                    inputs = plan.graph[workflows.CONDITIONING]["inputs"]
                    self.assertEqual(f"{inputs['width']}x{inputs['height']}", dimensions)

    def test_advertised_duration_boundary_builds_and_an_over_grid_value_is_rejected(self):
        maximum = self.capabilities["duration"]["maxSeconds"]
        request = self._request_for_mode("create", duration=maximum)
        plan = workflows.build(request, FULL, request.generation_mode)
        self.assertEqual(
            plan.graph[workflows.CONDITIONING]["inputs"]["length"],
            canvas.maximum_aligned_frame_count(),
        )
        with self.assertRaises(errors.PrivoraError):
            self._request_for_mode("create", duration=150)

    def test_advertised_reference_boundaries_and_fidelity_build(self):
        references = [
            {"type": "image", "role": "character", "id": f"i-{index}"}
            for index in range(self.capabilities["references"]["maxImages"])
        ] + [
            {"type": "video", "role": "motion", "id": f"v-{index}"}
            for index in range(self.capabilities["references"]["maxVideos"])
        ]
        request = request_module.parse({
            "mode": "references", "prompt": "x", "references": references,
        })
        self.assertEqual(request.references.file_count,
                         self.capabilities["references"]["maxTotal"])
        self.assertEqual(
            len(workflows.build(request, FULL).staging),
            self.capabilities["references"]["maxTotal"],
        )

        with self.assertRaises(errors.PrivoraError):
            request_module.parse({
                "mode": "references", "prompt": "x",
                "references": references + [{"type": "audio", "role": "music"}],
            })

        for fidelity in self.capabilities["references"]["fidelity"]:
            with self.subTest(fidelity=fidelity):
                parsed = self._request_for_mode("references", referenceFidelity=fidelity)
                plan = workflows.build(parsed, FULL)
                expected = "max" if fidelity == "high" else "match"
                self.assertEqual(
                    plan.graph[workflows.CONDITIONING]["inputs"]["ref_image_size"], expected
                )

    def test_advertised_generation_modes_match_each_family_executable_routes(self):
        expected_steps = {"quality": 20, "turbo": 8, "turboFast": 4}
        for family, mode in (
            (models.FL2VA, "create"), (models.REF2VA, "references")
        ):
            for wire_mode in self.capabilities["byFamily"][family]:
                with self.subTest(family=family, generation_mode=wire_mode):
                    parsed = self._request_for_mode(mode, generationMode=wire_mode)
                    plan = workflows.build(parsed, FULL, parsed.generation_mode)
                    self.assertEqual(plan.as_metadata()["steps"], expected_steps[wire_mode])

        self.assertNotIn("turbo", self.capabilities["byFamily"][models.REF2VA])
        ref_turbo = self._request_for_mode("references", generationMode="turbo")
        with self.assertRaises(errors.PrivoraError) as caught:
            workflows.build(ref_turbo, FULL, ref_turbo.generation_mode)
        self.assertEqual(caught.exception.code, errors.UNSUPPORTED_MODE)


class GraphStructureTests(unittest.TestCase):
    def test_both_families_share_the_sampler_to_save_tail(self):
        fl2va = workflows.build(create(), FULL, models.QUALITY).graph
        ref2va = workflows.build(references(), FULL, models.QUALITY).graph
        for node in ("noise", "guider", "sampler", "sigmas", "sample",
                     "decode_video", "decode_audio", "create_video", "save"):
            self.assertEqual(fl2va[node]["class_type"], ref2va[node]["class_type"], node)

    def test_the_canvas_reaches_the_conditioning_node(self):
        request = create(quality="hd", aspectRatio="16:9", duration=10)
        plan = workflows.build(request, FULL, models.QUALITY)
        inputs = plan.graph["conditioning"]["inputs"]
        self.assertEqual(inputs["width"], request.canvas.width)
        self.assertEqual(inputs["height"], request.canvas.height)
        self.assertEqual(inputs["length"], request.canvas.frames)

    def test_the_seed_reaches_the_noise_node(self):
        self.assertEqual(workflows.build(create(seed=4242), FULL, models.QUALITY)
                         .graph["noise"]["inputs"]["noise_seed"], 4242)

    def test_fidelity_reaches_the_reference_node(self):
        high = workflows.build(references(referenceFidelity="high"), FULL, models.QUALITY)
        self.assertEqual(high.graph["conditioning"]["inputs"]["ref_image_size"], "max")
        standard = workflows.build(references(), FULL, models.QUALITY)
        self.assertEqual(standard.graph["conditioning"]["inputs"]["ref_image_size"], "match")

    def test_keyframes_become_staged_loader_nodes(self):
        plan = workflows.build(
            request_module.parse({
                "mode": "animate", "prompt": "x", "seed": 1,
                "firstFrame": {"type": "image", "id": "a"},
                "lastFrame": {"type": "image", "id": "b"},
            }), FULL, models.QUALITY)
        self.assertIn("first_frame", plan.graph)
        self.assertIn("last_frame", plan.graph)
        self.assertEqual({s["node"] for s in plan.staging}, {"first_frame", "last_frame"})

    def test_reference_inputs_use_the_nodes_autogrow_names(self):
        plan = workflows.build(request_module.parse({
            "mode": "references", "prompt": "x", "seed": 1,
            "references": [
                {"type": "image", "role": "character"},
                {"type": "image", "role": "clothing"},
                {"type": "video", "role": "motion", "soundtrack": {"type": "audio", "role": "ambience"}},
                {"type": "audio", "role": "voice"},
            ],
        }), FULL, models.QUALITY)
        inputs = plan.graph["conditioning"]["inputs"]
        for name in ("ref_image_1", "ref_image_2", "ref_video_1", "ref_video_audio_1", "ref_audio_1"):
            self.assertIn(name, inputs, name)
        # The soundtrack is index-paired to its video, which is how the node pairs them.
        self.assertEqual(inputs["ref_video_audio_1"], ["ref_video_audio_1", 0])

    def test_every_staged_reference_has_a_loader_node(self):
        plan = workflows.build(request_module.parse({
            "mode": "references", "prompt": "x", "seed": 1,
            "references": [{"type": "image", "role": "character"},
                           {"type": "video", "role": "motion"},
                           {"type": "audio", "role": "music"}],
        }), FULL, models.QUALITY)
        for item in plan.staging:
            self.assertIn(item["node"], plan.graph)
            self.assertEqual(plan.graph[item["node"]]["inputs"][item["field"]], "")

    def test_the_graph_has_no_dangling_links(self):
        for request in (create(), references()):
            graph = workflows.build(request, FULL, models.TURBO_FAST if request.family == "fl2va"
                                    else models.QUALITY).graph
            for node_id, node in graph.items():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                        self.assertIn(str(value[0]), graph, f"{node_id} points at a missing node")


class GraphPrivacyTests(unittest.TestCase):
    """The graph carries the prompt by necessity. It must carry nothing else of the user's."""

    CANARY = "PROMPT_CANARY_PRIVORA_7F91A2"

    def test_the_output_filename_is_never_derived_from_user_content(self):
        plan = workflows.build(create(prompt=f"{self.CANARY} a scene"), FULL, models.QUALITY)
        prefix = plan.graph["save"]["inputs"]["filename_prefix"]
        self.assertEqual(prefix, "video/H3")
        self.assertNotIn(self.CANARY, prefix)

    def test_reference_loaders_start_empty_rather_than_carrying_a_caller_filename(self):
        plan = workflows.build(request_module.parse({
            "mode": "references", "prompt": "x", "seed": 1,
            "references": [{"type": "image", "role": "character",
                            "id": "../../etc/passwd", "url": "https://x/evil.png"}],
        }), FULL, models.QUALITY)
        # Nothing the caller supplied reaches the graph. The handler writes a generated
        # name into the slot after it has staged the file itself.
        self.assertEqual(plan.graph["ref_image_1"]["inputs"]["image"], "")
        self.assertNotIn("passwd", str(plan.graph))
        self.assertNotIn("evil.png", str(plan.graph))

    def test_the_prompt_appears_only_in_the_conditioning_node(self):
        plan = workflows.build(create(prompt=f"{self.CANARY} a scene"), FULL, models.QUALITY)
        carrying = [n for n, node in plan.graph.items() if self.CANARY in str(node)]
        self.assertEqual(carrying, ["conditioning"])

    def test_turbo_and_quality_share_one_output_boundary(self):
        # A faster path must not be a different privacy path.
        quality = workflows.build(create(), FULL, models.QUALITY).graph
        turbo = workflows.build(create(), FULL, models.TURBO_FAST).graph
        self.assertEqual(quality["save"], turbo["save"])
        self.assertEqual(quality["create_video"], turbo["create_video"])
        self.assertEqual(quality["decode_video"], turbo["decode_video"])


if __name__ == "__main__":
    unittest.main()
