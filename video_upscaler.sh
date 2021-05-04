


#Path where the initial video clips are located
path_to_clips="video"
mkdir -p "$path_to_clips/results"
output_path="$path_to_clips/results"

video_extention='mp4'
fps=30
audio_type='aac'
# Path of to be used a cache, be sure to have a fast read and write as well as some space
# using hard drives will slow the process.
cache_path='cache' # Folder must be empty !


#Loading model and weights
path_weights="model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"
model_type='UpConv_7'


# Batch size telling how many images you'll be processing simutanuously increase this as much as your VRAM allows it
batch_size=1
find $path_to_clips -iname "*.$video_extention"> $cache_path'/videos.txt'
readarray -t video_list < $cache_path'/videos.txt'

for vid in "${video_list[@]}"
do  

	raw_name=$(basename $vid)
    echo "Upscaling $raw_name"
	ffmpeg -i $vid -vn -acodec copy "$cache_path/output-audio.$audio_type"
	mkdir -p "$cache_path/frames/class_1"
	mkdir -p "$cache_path/results/"
	ffmpeg -i $vid "$cache_path/frames/class_1/thumb%010d.png" -hide_banner
	python3 video_up.py $path_to_clips $cache_path $model_type $path_weights $batch_size

	ffprobe -select_streams v -show_frames $vid  | grep pkt_duration_time= >"$cache_path/frames.txt"
	sed 's/pkt_duration_time=/duration /g' "$cache_path/frames.txt" > "$cache_path/frametimes.txt"
	ls --quoting-style=shell "$cache_path/results/" >"$cache_path/filenames.txt" && sed -i -e 's/^/file /' "$cache_path/filenames.txt"
	paste -d \\n "$cache_path/filenames.txt" "$cache_path/frametimes.txt" > "$cache_path/results/concatfile.txt"

	prev_path=$(pwd)
	cd "$cache_path/results/"
	ffmpeg -f concat -safe 0 -i concatfile.txt -threads 4 -vf fps=$fps "output.$video_extention" #-pix_fmt yuv420p
	cd $prev_path
	ffmpeg -i "$cache_path/results/output.$video_extention" -i "$cache_path/output-audio.$audio_type" -c copy -map 0:v:0 -map 1:a:0 "$output_path/$raw_name"
	rm -r "$cache_path/"*

done

