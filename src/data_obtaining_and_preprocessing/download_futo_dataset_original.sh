#!/usr/bin/env bash

SPLITS=(train dev test)
BASE_URL="https://huggingface.co/datasets/futo-org/swipe.futo.org/resolve/main"
DEST_DIR="./data/futo"

mkdir -p "$DEST_DIR"

for split in "${SPLITS[@]}"; do
	url="$BASE_URL/$split.jsonl"
	out="$DEST_DIR/$split.jsonl"

	if [[ -f "$out" ]]; then
		echo "Skipping $split (exists)."
		continue
	fi

	echo "Downloading $split → $out"
	tmp="$out.tmp"
	rm -f "$tmp"
	curl -fL "$url" -o "$tmp"
	mv -f "$tmp" "$out"
	echo "Saved: $out"
done

echo "All done. Files in: $DEST_DIR"