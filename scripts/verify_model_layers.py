#!/usr/bin/env python3
"""Structurally verify that the model layers landed on top of the code image.

Content integrity is already proven elsewhere: build_model_layer.py checks the exact byte
size and SHA-256 of every model *before* the layer is appended. What is left to confirm is
that all the expected layers actually made it onto the final manifest, in order.

A registry manifest reports **compressed** layer sizes, so the check is a per-layer
compression-ratio band against the known uncompressed size from models.tsv, not a raw
total. Quantised safetensors gzip to roughly 88-93% of their original size, so the band is
deliberately wide: tight enough to catch a truncated or misordered layer, loose enough to
survive a change in compression behaviour.

Usage:
    verify_model_layers.py --manifest final.json --code-manifest code.json --models models.tsv
"""

from __future__ import annotations

import argparse
import json
import sys

# Observed ratios on the real build were 0.880-0.932. The band allows anything from
# "compressed to half" to "slightly larger than raw" (gzip can add overhead on random data).
MIN_RATIO = 0.50
MAX_RATIO = 1.05


def load_models(path: str) -> list[tuple[str, str, int]]:
    models = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise SystemExit(f"models.tsv line has {len(fields)} fields, expected 5: {line!r}")
            name, _url, dest, size, _sha = fields
            models.append((name, dest, int(size)))
    return models


def layers_of(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "manifests" in manifest:
        raise SystemExit(
            f"{path} is a manifest index, not an image manifest. The code image must be a "
            "single-platform manifest (build with provenance: false) for crane append."
        )
    return manifest.get("layers", [])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="crane manifest of the final image")
    parser.add_argument("--code-manifest", required=True, help="crane manifest of the code image")
    parser.add_argument("--models", default="models.tsv")
    args = parser.parse_args(argv)

    models = load_models(args.models)
    final = layers_of(args.manifest)
    code = layers_of(args.code_manifest)

    print(f"code image layers:  {len(code)}")
    print(f"final image layers: {len(final)} (+{len(final) - len(code)})")
    print(f"expected models:    {len(models)}")

    appended = final[len(code):]
    if len(appended) != len(models):
        print(
            f"::error::Expected exactly {len(models)} appended model layers, found "
            f"{len(appended)}."
        )
        return 1

    # The code image's own layers must still be the prefix of the final image; if they are
    # not, crane appended onto the wrong base.
    for index, (code_layer, final_layer) in enumerate(zip(code, final)):
        if code_layer["digest"] != final_layer["digest"]:
            print(f"::error::Layer {index} differs from the code image; wrong base was used.")
            return 1

    failures = 0
    total_compressed = 0
    total_raw = 0

    print()
    print(f"{'model':<12} {'raw GB':>8} {'gz GB':>8} {'ratio':>7}  status")
    for (name, dest, raw_size), layer in zip(models, appended):
        compressed = layer["size"]
        ratio = compressed / raw_size
        total_compressed += compressed
        total_raw += raw_size

        ok = MIN_RATIO <= ratio <= MAX_RATIO
        status = "OK" if ok else f"OUT OF BAND [{MIN_RATIO}-{MAX_RATIO}]"
        print(
            f"{name:<12} {raw_size / 1e9:>8.2f} {compressed / 1e9:>8.2f} {ratio:>7.3f}  {status}"
        )
        if not ok:
            print(f"::error::Layer for {dest} is implausible: {compressed} compressed bytes "
                  f"for {raw_size} raw bytes (ratio {ratio:.3f}).")
            failures += 1

    print()
    print(f"Total raw {total_raw / 1e9:.2f} GB -> compressed {total_compressed / 1e9:.2f} GB "
          f"(overall ratio {total_compressed / total_raw:.3f})")

    if failures:
        return 1
    print(f"All {len(models)} model layers are present, in order, at plausible sizes.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
