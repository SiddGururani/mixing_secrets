import yaml
import re
import os, errno
import librosa
import numpy as np

def gen_yaml(directory, base_path, save_path, move_raw = False):
    artist, song, _ = directory.split('_')
    ID = '_'.join([artist, song])
    yaml_obj = init_medley_yaml()
    yaml_obj['artist'] = artist
    yaml_obj['title'] = song
    yaml_obj['has_bleed'] = 'no'
    yaml_obj['instrumental'] = 'no'
    yaml_obj['mix_filename'] = ID+'_MIX.wav'
    yaml_obj['origin'] = 'Mixing Secrets'
    yaml_obj['raw_dir'] = ID+'_RAW'
    yaml_obj['stem_dir'] = ID+'_STEMS'
    yaml_obj['version'] = '3.0'
    make_dir(os.path.join(save_path, ID))
    make_dir(os.path.join(save_path, ID, ID+'_RAW'))
    make_dir(os.path.join(save_path, ID, ID+'_STEMS'))
    if os.path.isfile(os.path.join(save_path, ID, ID+'_METADATA.yaml')):
        # Write code here to fix the drum tracks by adding the room mic to the drum stem.
        print('Metadata exists')
        return
    
    # Get all track paths
    all_tracks = os.listdir(os.path.join(base_path, directory))
    all_tracks = [os.path.join(base_path, directory, track) for track in all_tracks if track.endswith('.wav')]
    
    # Get drum, synth, loops and sfx tracks
    raw_drums = find_drum_tracks(os.path.join(base_path, directory))
    raw_loops = find_loop_tracks(os.path.join(base_path, directory))
    raw_synths = find_synth_tracks(os.path.join(base_path, directory))
    raw_sfx = find_sfx_tracks(os.path.join(base_path, directory))
    
    # Find remaining tracks
    used_tracks = []
    used_tracks.extend(raw_drums + raw_synths + raw_sfx + raw_loops)
    rem_tracks = list(set(all_tracks)^set(used_tracks))
    
    # Make stems for drums, sfx, loops and synths
    make_stem(yaml_obj, os.path.join(save_path, ID, ID+'_STEMS'), raw_drums, 'drum set', ID+'_STEM_drums.wav')
    make_stem(yaml_obj, os.path.join(save_path, ID, ID+'_STEMS'), raw_loops, 'drum machine', ID+'_STEM_loops.wav')
    make_stem(yaml_obj, os.path.join(save_path, ID, ID+'_STEMS'), raw_synths, 'synthesizer', ID+'_STEM_synth.wav')
    make_stem(yaml_obj, os.path.join(save_path, ID, ID+'_STEMS'), raw_sfx, 'fx/processed sound', ID+'_STEM_sfx.wav')
    
    # Add remaining tracks to yaml as a stem with 1 raw track. Manually fix this later.
    add_rem_tracks(yaml_obj, os.path.join(save_path, ID, ID+'_STEMS'), rem_tracks)
    
    # Move all raw files to RAW folder. Default False
    if move_raw == True:
        move_raw_tracks(all_tracks, os.path.join(save_path, ID, ID+'_RAW'))
    
    # Write YAML
    f = open(os.path.join(save_path, ID, ID+'_METADATA.yaml'),'w')
    yaml.dump(yaml_obj, f, default_flow_style=False)
    f.close()
    
def init_medley_yaml():
    object = {}
    object['album'] = ''
    object['artist'] = ''
    object['composer'] = ''
    object['excerpt'] = ''
    object['genre'] = ''
    object['has_bleed'] = ''
    object['instrumental'] = ''
    object['mix_filename'] = ''
    object['origin'] = ''
    object['producer'] = ''
    object['raw_dir'] = ''
    object['stem_dir'] = ''
    object['stems'] = {}
    object['title'] = ''
    object['version'] = ''
    object['website'] = ''
    return object
    
def find_drum_tracks(base_path):
    all_raw = os.listdir(base_path)
    drums = ['kick', 'snare', 'overhead', 'tom', 'drum', 'hat', 'cymbal', 'sharedown', 'ride', 'crash', 'cowbell', 'sticks']
    drum_tracks = set([os.path.join(base_path, track) for track in all_raw for drum in drums if drum in track.lower() and track.endswith('.wav')])
    return list(drum_tracks)


def find_synth_tracks(base_path):
    all_raw = os.listdir(base_path)
    synth_tracks = set([os.path.join(base_path, track) for track in all_raw if 'synth' in track.lower() and track.endswith('.wav')])
    return list(synth_tracks)


def find_loop_tracks(base_path):
    all_raw = os.listdir(base_path)
    loop_tracks = set([os.path.join(base_path, track) for track in all_raw if 'loop' in track.lower() and track.endswith('.wav')])
    return list(loop_tracks)

def find_sfx_tracks(base_path):
    all_raw = os.listdir(base_path)
    sfx_tracks = set([os.path.join(base_path, track) for track in all_raw if 'sfx' in track.lower() and track.endswith('.wav')])
    return list(sfx_tracks)

def make_stem(obj, stems_path, tracks, inst_name, file_name):
    if len(tracks) == 0:
        print('Empty track list sent for stem creation')
        return
    y, sr = librosa.load(tracks[0], sr=None)
    for i in range(len(tracks) - 1):
        y_add = librosa.load(tracks[i+1], sr=None)[0]
        l = len(y)
        l_add = len(y_add)
        if l > l_add:
            y_add = np.pad(y_add, (0, l - l_add), 'constant')
        elif l < l_add:
            y = np.pad(y, (0, l_add - l), 'constant')
        y += y_add
    y = y/len(tracks)
    path_to_write = os.path.join(stems_path, file_name)
    librosa.output.write_wav(path_to_write, y, sr)
    # Add stem to yaml object
    count = len(obj['stems'])
    if count+1 < 10:
        count = '0'+str(count+1)
    else:
        count = str(count+1)
    obj['stems']['S'+count] = {}
    obj['stems']['S'+count]['component'] = ''
    obj['stems']['S'+count]['filename'] = file_name
    obj['stems']['S'+count]['instrument'] = inst_name
    obj['stems']['S'+count]['raw'] = {}
    for i, track in enumerate(tracks):
        track_name = os.path.split(track)[1]
        if i < 10:
            raw_count = '0'+str(i+1)
        else:
            raw_count = str(i+1)
        obj['stems']['S'+count]['raw']['R'+raw_count] = {}
        obj['stems']['S'+count]['raw']['R'+raw_count]['filename'] = track_name
        obj['stems']['S'+count]['raw']['R'+raw_count]['instrument'] = get_instrument_from_track_name(track_name)
        
def add_rem_tracks(obj, save_path, rem_tracks):
    for i, track in enumerate(rem_tracks):
        track_name = os.path.split(track)[1]
        inst_name = get_instrument_from_track_name(track_name)
        make_stem(obj, save_path, [track], inst_name, track_name)
        
        
def get_instrument_from_track_name(track_name):
    track_name = track_name.strip('.wav')
    regex = r"(\d*_)([a-zA-Z\D]*)"
    match = re.findall(regex, track_name)
    inst_name = '_'.join([x for (_,x) in match])
    return inst_name
        
def find_all_instruments(base_path):
    instruments = set()
    regex = r"(\d*_)([a-zA-Z\D]*)"
    for x in os.listdir(base_path):
        for track in os.listdir(os.path.join(base_path, x)):
            if track.endswith(".wav"):
                try:
                    track_name = track.strip('.wav')
                    match = re.findall(regex, track_name)
                    inst_name = '_'.join([x for (_,x) in match])
                    instruments.add(inst_name)
                except:
                    print(track)
    return instruments

def move_raw_tracks(tracks, destination):
    for track in tracks:
        track_name = os.path.split(track)[1]
        os.rename(track, os.path.join(destination, track_name))

def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
base_path = '/home/sgururani/Mixing_Secrets/Unzipped' # Path to unzipped clean archives
save_path = '/home/sgururani/Mixing_Secrets/Medley_Format/Audio' # Path to MedleyDB Audio Directory

problematic = [109,110, 125, 144, 200, 227]
for i, directory in enumerate(os.listdir(base_path)):
    print(i, directory)
    if i in problematic:
        print('Problematic')
        continue
    gen_yaml(directory, base_path, save_path)