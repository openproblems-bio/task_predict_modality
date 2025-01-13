#/bin/bash

res=/opt

## VIASH START
meta_resources_dir=src/shuji_suzuki
res=src/shuji_suzuki/resources
## VIASH END

repo="$res/shu65_openproblems"

echo "Copying data to output"
cp -r "$par_input" "$par_output"

echo "Making compressed dataset"
python3 "$repo/script/make_compressed_dataset.py" --data_dir "$par_output"

echo "Making additional files"
python3 "$repo/script/make_additional_files.py" --data_dir "$par_output"

echo "Making cite input mask"
python3 "$repo/script/make_cite_input_mask.py" \
  --data_dir "$par_output" \
  --hgnc_complete_set_path "$res/hgnc_complete_set.txt" \
  --reactome_pathways_path "$res/ReactomePathways.gmt"

echo "Training multi"
python3 "$repo/scripts/train_mode.py" --data_dir "$par_output" --task_type multi

echo "Training cite"
python3 "$repo/scripts/train_mode.py" --data_dir "$par_output" --task_type cite
