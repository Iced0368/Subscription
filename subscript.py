import whisper
import torch
import numpy as np
from whisper.utils import WriteSRT
from googletrans import Translator
from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
import pygame


from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.0-Q16\\magick.exe"})

model = whisper.load_model('medium')
model.eval()

def get_text_width(font, fontsize, text):
    font = pygame.font.Font(font, fontsize)
    text_surface = font.render(text, True, (0, 0, 0))
    return text_surface.get_width()

def str_to_rgb(color_name):
    color_name = color_name.lower()
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
    }
    if color_name in color_map:
        return np.array(color_map[color_name])
    else:
        return np.array(color_map['black'])
        


def second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)

    return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)

def print_scripts(script):
    for segment in script['segments']:
        start = second_to_timecode(segment['start'])
        end = second_to_timecode(segment['end'])
        print("[{} --> {}]".format(start, end), end="")
        print(segment['text'])

def newlined_text(scripts):
    result = ''
    for c in scripts['text']:
        result += c
        if c in '.!?':
            result += '\n'
    return result

def write_srt(scripts, filename):
    with open(os.path.join(filename + ".srt"), "w") as srt:
        WriteSRT(scripts["segments"])


def transcribe(filename):
    with torch.no_grad():
        return model.transcribe(filename, no_speech_threshold=0.5)
    
def translate_script(scripts, language='ko'):
    result = scripts.copy()
    translator = Translator()
    for segment in result['segments']:
        segment['text'] = ' ' + translator.translate(segment['text'], dest=language).text
    result['text'] = " ".join([segment['text'] for segment in result['segments']])
    return result



def simplify_segments(segments):
    seg = []
    for segment in segments:
        seg.append(((segment['start'], segment['end']), segment['text']))
    return seg


def add_subtitles(filename, subtitle, output_filename, fontsize=36, color='white', bg_color='transparent'):
    subs = simplify_segments(subtitle)
    clip = VideoFileClip(filename)
    font = 'font/NanumSquareB.ttf'
    generator = lambda txt: TextClip(
        txt.strip(), 
        method='caption', 
        size=(min(int(clip.size[0]*0.66), get_text_width(font, fontsize, txt)), None), 
        font=font, 
        fontsize=fontsize, 
        color=color, 
        bg_color=bg_color,
        align='center'
    ).margin(fontsize//6, color=str_to_rgb(bg_color), opacity=0.0 if bg_color=='transparent' else 1.0)
    
    subtitles = SubtitlesClip(subs, generator)
    subtitles = subtitles.subclip(0, clip.duration)

    video = CompositeVideoClip([clip, subtitles.set_position(('center', 0.9), relative=True)])
    video.write_videofile(output_filename)



filename = "example4.mp4"
scripts = transcribe(filename)
scripts = translate_script(scripts, 'ko')
#write_srt(scripts, filename)

add_subtitles(filename, scripts['segments'], "example_subtitled.mp4", fontsize=24, color='white', bg_color='black')
#print_scripts(scripts)
print(newlined_text(scripts))