"""Security and validation tests for user-supplied reference video and audio.

Run with:  python -m unittest discover -s tests -v

ffprobe is injected rather than invoked, so these run anywhere. What is being tested is the
decision-making around the probe - what gets rejected, when, and whether the rejection says
anything it should not - rather than ffmpeg itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privora import errors, media  # noqa: E402

CANARY = "REFERENCE_CANARY_PRIVORA_8B62C4"


def probe_payload(kind="video", *, codec=None, duration="8.0", width=1920, height=1080,
                  channels=2, extra_streams=0):
    stream = {"codec_type": kind, "duration": duration}
    if kind == "video":
        stream.update(codec_name=codec or "h264", width=width, height=height,
                      avg_frame_rate="24/1")
    else:
        stream.update(codec_name=codec or "aac", sample_rate="48000", channels=channels)
    streams = [stream] + [{"codec_type": "data"} for _ in range(extra_streams)]
    return {"streams": streams, "format": {"duration": duration}}


def runner_returning(payload):
    return lambda command: payload


class ContainerSniffTests(unittest.TestCase):
    def test_known_video_containers_are_recognised(self):
        self.assertEqual(media.sniff(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 16, "video"), "mp4")
        self.assertEqual(media.sniff(b"\x1a\x45\xdf\xa3" + b"\x00" * 28, "video"), "webm")

    def test_known_audio_containers_are_recognised(self):
        self.assertEqual(media.sniff(b"RIFF____WAVE" + b"\x00" * 20, "audio"), "wav")
        self.assertEqual(media.sniff(b"fLaC" + b"\x00" * 28, "audio"), "flac")
        self.assertEqual(media.sniff(b"OggS" + b"\x00" * 28, "audio"), "ogg")

    def test_a_mislabelled_upload_is_refused_before_ffprobe(self):
        # A ZIP claiming to be a video should be turned away by a byte comparison, not by
        # a demuxer that has already been handed the file.
        with self.assertRaises(errors.PrivoraError) as caught:
            media.sniff(b"PK\x03\x04" + b"\x00" * 28, "video")
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)

    def test_a_truncated_header_does_not_raise_an_unexpected_error(self):
        with self.assertRaises(errors.PrivoraError):
            media.sniff(b"\x00", "audio")


class SizeLimitTests(unittest.TestCase):
    def test_an_empty_file_is_refused(self):
        with self.assertRaises(errors.PrivoraError):
            media.check_size(b"", "video")

    def test_an_oversized_file_is_refused_with_the_limit(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            media.check_size(b"x" * (media.MAX_VIDEO_BYTES + 1), "video")
        self.assertEqual(caught.exception.details["limitBytes"], media.MAX_VIDEO_BYTES)

    def test_audio_has_its_own_smaller_ceiling(self):
        self.assertLess(media.MAX_AUDIO_BYTES, media.MAX_VIDEO_BYTES)
        media.check_size(b"x" * 1024, "audio")


class ProbeTests(unittest.TestCase):
    def test_a_normal_video_probes_cleanly(self):
        info = media.probe("/tmp/x.mp4", "video", runner=runner_returning(probe_payload()))
        self.assertEqual(info.codec, "h264")
        self.assertEqual((info.width, info.height), (1920, 1080))
        self.assertAlmostEqual(info.duration_seconds, 8.0)
        self.assertAlmostEqual(info.frame_rate, 24.0)

    def test_a_normal_audio_probes_cleanly(self):
        info = media.probe("/tmp/x.wav", "audio",
                           runner=runner_returning(probe_payload("audio")))
        self.assertEqual(info.sample_rate, 48000)
        self.assertEqual(info.channels, 2)

    def test_the_command_never_uses_a_shell_and_never_interpolates_the_path(self):
        seen = {}

        def capture(command):
            seen["command"] = command
            return probe_payload()

        hostile = "/tmp/$(rm -rf ~).mp4"
        media.probe(hostile, "video", runner=capture)
        self.assertIsInstance(seen["command"], list, "an array, never a shell string")
        self.assertEqual(seen["command"][-1], hostile, "the path is a discrete argument")
        self.assertNotIn(hostile, " ".join(seen["command"][:-1]))

    def test_an_unsupported_codec_is_refused(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            media.probe("/tmp/x.mp4", "video",
                        runner=runner_returning(probe_payload(codec="cinepak")))
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)
        self.assertEqual(caught.exception.details["codec"], "cinepak")

    def test_an_enormous_video_is_refused(self):
        with self.assertRaises(errors.PrivoraError):
            media.probe("/tmp/x.mp4", "video",
                        runner=runner_returning(probe_payload(width=16384, height=8640)))

    def test_a_file_with_no_matching_stream_is_refused(self):
        with self.assertRaises(errors.PrivoraError) as caught:
            media.probe("/tmp/x.mp4", "video",
                        runner=runner_returning(probe_payload("audio")))
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_TYPE)

    def test_a_stream_bomb_is_refused(self):
        with self.assertRaises(errors.PrivoraError):
            media.probe("/tmp/x.mp4", "video",
                        runner=runner_returning(probe_payload(extra_streams=50)))

    def test_a_file_with_no_duration_is_refused(self):
        payload = probe_payload()
        payload["streams"][0]["duration"] = None
        payload["format"]["duration"] = None
        with self.assertRaises(errors.PrivoraError) as caught:
            media.probe("/tmp/x.mp4", "video", runner=runner_returning(payload))
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_DURATION)

    def test_a_probe_timeout_becomes_a_clean_rejection(self):
        def timeout(command):
            raise subprocess.TimeoutExpired(command, media.PROBE_TIMEOUT_SECONDS)

        with self.assertRaises(errors.PrivoraError) as caught:
            media.probe("/tmp/x.mp4", "video", runner=timeout)
        self.assertEqual(caught.exception.code, errors.REFERENCE_PREPROCESSING_FAILED)

    def test_a_crashing_probe_becomes_a_clean_rejection(self):
        def explode(command):
            raise RuntimeError("ffprobe exited 139")

        with self.assertRaises(errors.PrivoraError) as caught:
            media.probe("/tmp/x.mp4", "video", runner=explode)
        self.assertEqual(caught.exception.code, errors.REFERENCE_PREPROCESSING_FAILED)


class AggregateDurationTests(unittest.TestCase):
    def test_individually_legal_files_can_still_be_collectively_too_long(self):
        infos = [media.MediaInfo("video", 15.0, "h264") for _ in range(3)]
        media.check_aggregate(infos, "video", limit_seconds=media.MAX_TOTAL_VIDEO_SECONDS)

        infos.append(media.MediaInfo("video", 15.0, "h264"))
        with self.assertRaises(errors.PrivoraError) as caught:
            media.check_aggregate(infos, "video", limit_seconds=media.MAX_TOTAL_VIDEO_SECONDS)
        self.assertEqual(caught.exception.code, errors.INVALID_REFERENCE_DURATION)
        self.assertEqual(caught.exception.details["totalSeconds"], 60.0)

    def test_the_aggregate_ceiling_follows_the_per_type_limits(self):
        self.assertEqual(media.MAX_TOTAL_VIDEO_SECONDS, 45.0)   # 15s x 3 videos
        self.assertEqual(media.MAX_TOTAL_AUDIO_SECONDS, 45.0)   # 15s x 3 clips


class RejectionPrivacyTests(unittest.TestCase):
    """A rejection describes shape, never content or provenance."""

    def test_no_rejection_carries_the_file_path(self):
        path = f"/tmp/privora/job/{CANARY}.mp4"
        for runner in (runner_returning(probe_payload(codec="cinepak")),
                       runner_returning(probe_payload(width=16384))):
            with self.assertRaises(errors.PrivoraError) as caught:
                media.probe(path, "video", runner=runner)
            self.assertNotIn(CANARY, caught.exception.message)
            self.assertNotIn(CANARY, str(caught.exception.details))

    def test_an_internal_diagnostic_may_name_the_path_but_is_never_returned(self):
        path = f"/tmp/privora/job/{CANARY}.mp4"

        def timeout(command):
            raise subprocess.TimeoutExpired(command, 1)

        with self.assertRaises(errors.PrivoraError) as caught:
            media.probe(path, "video", runner=timeout)
        self.assertNotIn(CANARY, str(caught.exception.as_response()))
        self.assertIn(CANARY, caught.exception.as_log_line())

    def test_the_metadata_reports_shape_only(self):
        info = media.probe("/tmp/x.mp4", "video", runner=runner_returning(probe_payload()))
        metadata = info.as_metadata()
        self.assertEqual(set(metadata), {"durationSeconds", "codec", "width", "height", "frameRate"})


if __name__ == "__main__":
    unittest.main()
