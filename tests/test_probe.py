"""
Unit tests for probe, stream selection, HDR10 determination, and output naming.
"""

from pathlib import Path
import pytest
from av1_migrator.models import AudioStream, VideoStream
from av1_migrator.probe import (
    generate_output_paths,
    is_commentary_stream,
    is_english_language,
    is_hdr_metadata,
    select_audio_streams,
    select_video_stream,
)


def test_hdr_detection():
    # SMPTE 2084 / BT.2020 should be detected as HDR10
    assert is_hdr_metadata(color_transfer="smpte2084", color_primaries="bt2020", color_space="bt2020nc") is True
    # BT.709 should be SDR
    assert is_hdr_metadata(color_transfer="bt709", color_primaries="bt709", color_space="bt709") is False


def test_generate_output_paths_no_duplicate_tags():
    src = Path("/movies/Harry Potter (2001).mkv")
    final, temp = generate_output_paths(src, is_hdr=True, cq=28, container="mkv")
    assert final.name == "Harry Potter (2001) [AV1 1080p HDR10 CQ28].mkv"
    assert temp.name == "Harry Potter (2001) [AV1 1080p HDR10 CQ28].mkv.encoding.mkv"

    # If filename already had tag, do not duplicate
    src_tagged = Path("/movies/Harry Potter (2001) [AV1 1080p HDR10 CQ28].mkv")
    final2, temp2 = generate_output_paths(src_tagged, is_hdr=True, cq=28, container="mkv")
    assert final2.name == "Harry Potter (2001) [AV1 1080p HDR10 CQ28].mkv"


def test_select_video_stream_multi_stream():
    v1 = VideoStream(index=0, codec_name="hevc", width=3840, height=2160, is_default=False, duration=100.0)
    v2 = VideoStream(index=1, codec_name="hevc", width=1920, height=1080, is_default=True, duration=100.0)

    # Highest resolution preferred by default
    selected = select_video_stream([v1, v2], prefer_native_1080p=False)
    assert selected is not None
    assert selected.index == 0
    assert selected.width == 3840

    # If prefer_native_1080p is true
    selected_1080 = select_video_stream([v1, v2], prefer_native_1080p=True)
    assert selected_1080 is not None
    assert selected_1080.index == 1
    assert selected_1080.width == 1920


def test_select_audio_streams_ranking_and_commentary():
    a_comm = AudioStream(index=1, codec_name="aac", language="eng", title="Director Commentary", is_commentary=True)
    a_truehd = AudioStream(index=2, codec_name="truehd", language="eng", title="Surround 7.1", channels=8, is_default=True)
    a_ac3 = AudioStream(index=3, codec_name="ac3", language="eng", title="Stereo", channels=2)
    a_fre = AudioStream(index=4, codec_name="dts", language="fre", title="French")

    selected = select_audio_streams(
        [a_comm, a_truehd, a_ac3, a_fre],
        accepted_languages=["eng", "en"],
        exclude_commentary=True,
    )

    # Should filter out french and commentary, ranking TrueHD above AC3
    assert len(selected) == 2
    assert selected[0].index == 2
    assert selected[0].codec_name == "truehd"
    assert selected[1].index == 3
