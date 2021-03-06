# Upscale
## Introduction
This work is derivated from the work of https://github.com/yu45020/Waifu2x that contains the main idea and the scripts I updated. you can find that I kept the models, the trained weights and lots of classes that were not made from me. However I improved the implementation to speed up the upscaling process compared to video2x that processes one image at a time we offer here a way to process as many images as your VRAM allows it: make some test and this will significantly speed up the process depending on the hardware that you have.
My goal here is to suggest a methodology to upscale your movie so that you can enjoy it at a higher quality especially if the higher resolution is not available, of course you can run these programs on any computer but without a cuda capable gpu you are going to wait a lot of time.

## Requirements
Running this program can not be made out of the box and requires some piece of hardware/software

* You need ffmpeg for dealing with the videos and extract the audio, the frames and build them back together afterwards
* You need [pytorch](https://pytorch.org/), if you don't have a cuda capable GPU, you can use the CPU version but it will be very slow
* (optionnal but very advised): a CUDA capable GPU with CUDA and Cudnn installed on it, if you have no idea on how to install all this, [here is a guide on how to install a linux environement with pytorch, cuda and cudnn](https://github.com/williampiat3/DeepLearningOnGamingLaptop)
* Depending on the movie you are upscaling you might need a lot of free space on your disk: 1 free Tb is a good start


## Principle
### What we suggest
The goal here is to increase the quality of a movie and therefore we will increase the quality of its frames and keeping the audio, the subtitles so here is how we are going to do it: First we will extract the audio, the subtitles and the frames from the movie, Then we will upscale the frames so as to increase teh quality of the movie, and finally we will merge all the upscaled frames back with the audio and the subtitles

### How is this different from other solutions
## On inference
THere are mainly two differences between our solution and other upscaling softwares:
* We offer the possiblity to stop the upscaling whenever you want, as this is the longest operation you might want to have it running during the night or during free time but you might want to use the computer at some point for your daily use, we provide utility scripts in order to stop the calculation and resume where it left.
* On the other hand we process multiple frames at a times and use a multiprocessed reading and writting of the images, this might seem easy but the way the upscaling operation works makes it rather difficult to do therefore we see some major improvment in the upscaling process especially if you have a lot of VRAM. 

To put it shortly the images are cut into smaller patches of 70 pixels x 70 pixels before being processed through the network however not all the widths and heights are dividable by 70 therefore some patches are not 70x70 but smaller depending on which part of the frame they are located, Thus discarding the possibility to parallelize the processing of the patches of 1 images: But since the frames that you are using come from the same movie they will have the same size and therefore the same patches size. Here is a small example to show you the problem of parallelization with 3x3 patches on a 10x8 image and how we tackled it for a movie
<p align="center">
 <img src="./illustrations/Capture.PNG" width=49% height=49%>
 <img src="./illustrations/Capture2.PNG" width=49% height=49%>
</p>
This allows to process much faster the frames than the current solutions

The previous solution was concatenating the patches after being processed resulting in, sometimes, noise on the borders of the patches, we decides to average out the overlapping patches in order to reduce border effects.

## On training
Our contribution on the training is the following we added a compression noise following the original waifu2x method this allows the network to remove the compression noise that all movies/cartoons have. Here we present strong compression on a low resolution image:
<p align="center">
 <img src="./trainings/lr.png" width=49% height=49%>
 <img src="./illustrations/test.png" width=49% height=49%>

</p>
We are not imposing one single level of noise, we are compressing the image in a random interval of quality range, this had a very positive effect on the output however the result can still look different than the high quality base


We added a GAN so as to improve the look of the result, this strategy is not new (see [this paper for instance](https://arxiv.org/abs/1609.04802)) however contrary to this paper we decided to go for a Wasserstein GAN that allows a better feedback from the Discriminant. Here is the processing graph to give you the idea of the process. It is a rather common architecture this is why we will not dwell too much into it
<p align="center">
 <img src="./illustrations/gan.png">
</p>
For the Discriminant to be able to operate on a great range of images we decided to make it take 64x64 pixels patches this is why there are extra steps in the code to fold the images


## Steps

### Remove audio and subtitles
Use ffmpeg to extract the audio :
```
ffmpeg -i input-video.mp4 -vn -acodec copy output-audio.ogg
```
Audio are not always in .ogg, be sure the check all streams

And the subtitles, in case your file is an mkv
```
ffmpeg -i input-video.mkv -map 0:s:0 subs.srt
```
If you have multiple audio and sutitles in your file you will have to extract all the streams individually. You can plot all streams by using the following code:
```
video1='path/to/your/video.mp4'
ffprobe -show_entries stream=index,codec_type:stream_tags=language -of compact $video1 2>&1 | { while read line; do if $(echo "$line" | grep -q -i "stream #"); then echo "$line"; fi; done; while read -d $'\x0D' line; do if $(echo "$line" | grep -q "time="); then echo "$line" | awk '{ printf "%s\r", $8 }'; fi; done;}
```

### Extract all frames
Create a folder 'frames' and another inside it called 'class_1' where to put all the frames as there will be a lot of them, then run the following command:
```
ffmpeg -i input-video.mp4 frames/class_1/thumb%010d.png -hide_banner
```
This can take a while and this task is a CPU heavy task so make sure your PC is not in a overheated place, your fans should be enough for cooling it

### Upscale
Creare a folder that will recieve the upscaled images 'output' then open the test.py file in this github, change the input_path to specify the folder with the images, the output_path from the path of the folder where you'll be writting the images upscaled and the path of the weights of the model if you upscale images or cartoons, we recommand using locations on an SSD so as to improve I/O speed. You need to set the batch_size as high as your VRAM allows it, it will speed up the upscaling process a lot.

Then run the test.py program, this might take a while. The program writes every processed frames, this can give you a chance to see the overall progression or to stop the program and restart later.

### Stop the process and resume
This is optionnal but in order to stop the program and resume the computation where it stopped you simply have to move the images already done out of the 'frames/class_1' folder, we provide a python script called 'move_files.py' that checks the images already processed in the output folder and move the ones present in the input folder to the location you want, and then if you restart the test.py script it will resume where it left

### Build the video back and link the audio and subtitles
Find the framerate of the initial file:
```
ffmpeg -i filename
```
On the video stream you can have the fps (frame per second) you will need this information when building the video again

Once all frames were upscaled go to the output folder and use ffmeg to link them altogether in a movie and add the audio and subtitle streams, make sure to put the appropriate framerate
```
ffmpeg -framerate 24 -start_number 1 -i thumb%010d.png ../output.mp4
```
Note this method only works if you have a constant framerate for your initial video. Unconsistant framerate can lead to desynchronised audio.
To avoid this you can get every individual frame times and merge them similary to the initial file:
```
# Get all frame individual time:
ffprobe -select_streams v -show_frames input-video.mp4 | grep pkt_duration_time= >frames.txt  
# Turn them into usable data by ffmpeg
sed 's/pkt_duration_time=/duration /g' frames.txt >frametimes.txt

# Print a list of your upscaled frames located in 'folder/' and modify the list
# to be the format FFmpeg's concat filter requires:
ls --quoting-style=shell folder/ >filenames.txt && sed -i -e 's/^/file /' filenames.txt
# Be sure that only the Upscaled frames are in `folder/` otherwise this will
# insert unwanted names in the list created

# Combine your frame-time file and filenames file to a file that FFmpeg concat accepts:
paste -d \\n filenames.txt frametimes.txt > folder/concatfile.txt

# Then just put yourself in the folder with all frames and merge images:
cd folder/
ffmpeg -f concat -safe 0 -i concatfile.txt ../output.mp4
```
There are few arguments you might use for the last command:
* `ffmpeg -f concat -i concatfile.txt -pix_fmt yuv420p output.mp4` convert RGB to Y'UV table (increase computation time)
* `ffmpeg -f concat -i concatfile.txt -vf fps=30 output.mp4` generate 30 constant fps, the default value been 25. Be aware that it most likely uses interpolation so just stay close to the original video, otherwise you will lose frame or dupplicate to much data
* `ffmpeg -f concat -i concatfile.txt -threads 4 output.mp4` specify on how much thread you want to perform the work. Here 4 but keep it close to your number of core processor.

Once this this done, move `output.mp4` to the same file as `output-audio.ogg`

Add the audio 
```
ffmpeg -i output.mp4 -i output-audio.ogg -c copy -map 0:v:0 -map 1:a:0 output_with_sound.mp4
```

If your file is a mkv file you can add the subtitles you removed before
```
ffmpeg -i output_with_sound.mkv -i subs.srt -map 0 -map 1:s:0 output_with_subtitles.mkv
```
And that's it, a bit tedious but much more flexible than a software that you can run but it runs much faster, on a Geforce 1050 Ti Max Q it ran 3 times faster than video2X which is not neglictable when you are talking about hours or days of computation.

## Conclusion
We provide here a more 'handcrafted' version of upscaling but it is customisable to any model, any movie and can be stopped and resumed at any time it is therefore a good solution to anyone who doesn't want to lock its computer for days and that want to enjoy a better quality on old movies or cartoons. We e provide a worflow that is not that perfect but makes you understand all the steps that are necessary to upscale a video.
 You can check out our examples in the folders frames and results
<p align="center">
 <img src="./illustrations/comparison.png">
</p>

