#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:01:00 2021

@author: biren
"""

import os


seq_info_sdr = { 
                # Class A1 4k
                'Campfire': {'file-name':'Campfire_3840x2160_30fps_bt709_420_videoRange', 'cfg-name': 'Campfire.cfg'}, 
                'FoodMarket4' : {'file-name':'FoodMarket4_3840x2160_60fps_10bit_420', 'cfg-name': 'FoodMarket4.cfg'},
                'Tango2': {'file-name':'Tango2_3840x2160_60fps_10bit_420', 'cfg-name': 'Tango2.cfg'},
                
                # Class A2 4k
                'CatRobot': {'file-name': 'CatRobot_3840x2160_60fps_10bit_420_jvet', 'cfg-name': 'CatRobot.cfg'},
                'DaylightRoad2': {'file-name': 'DaylightRoad2_3840x2160_60fps_10bit_420', 'cfg-name': 'DaylightRoad2.cfg'},
                'ParkRunning3': {'file-name': 'ParkRunning3_3840x2160_50fps_10bit_420', 'cfg-name': 'ParkRunning3.cfg'},
                
                # Class B 1080p
                'MarketPlace': {'file-name': 'MarketPlace_1920x1080_60fps_10bit_420', 'cfg-name': 'MarketPlace.cfg'},
                'RitualDance': {'file-name': 'RitualDance_1920x1080_60fps_10bit_420', 'cfg-name': 'RitualDance.cfg'},
                'Cactus': {'file-name': 'Cactus_1920x1080_50', 'cfg-name': 'Cactus.cfg'},
                'BasketballDrive': {'file-name': 'BasketballDrive_1920x1080_50', 'cfg-name': 'BasketballDrive.cfg'},
                'BQTerrace': {'file-name': 'BQTerrace_1920x1080_60', 'cfg-name': 'BQTerrace.cfg'},
                
                # Class C WVGA
                'BasketballDrill': {'file-name': 'BasketballDrill_832x480_50', 'cfg-name': 'BasketballDrill.cfg'},
                'BQMall': {'file-name': 'BQMall_832x480_60', 'cfg-name': 'BQMall.cfg'},
                'PartyScene': {'file-name': 'PartyScene_832x480_50', 'cfg-name': 'PartyScene.cfg'},
                'RaceHorsesCC': {'file-name': 'RaceHorses_832x480_30', 'cfg-name': 'RaceHorsesC.cfg'},
                
                # Class D WOVGA
                'BasketballPass': {'file-name': 'BasketballPass_416x240_50', 'cfg-name': 'BasketballPass.cfg'},
                'BQSquare': {'file-name': 'BQSquare_416x240_60', 'cfg-name': 'BQSquare.cfg'},
                'BlowingBubbles': {'file-name': 'BlowingBubbles_416x240_50', 'cfg-name': 'BlowingBubbles.cfg'},
                'RaceHorsesCD': {'file-name': 'RaceHorses_416x240_30', 'cfg-name': 'RaceHorses.cfg'},
                
                # Class E 720p
                'FourPeople': {'file-name': 'FourPeople_1280x720_60', 'cfg-name': 'FourPeople.cfg'},
                'Johnny': {'file-name': 'Johnny_1280x720_60', 'cfg-name': 'Johnny.cfg'},
                'KristenAndSara': {'file-name': 'KristenAndSara_1280x720_60', 'cfg-name': 'KristenAndSara.cfg'},
                
                # Class F
                'BasketballDrillText': {'file-name': 'BasketballDrillText_832x480_50', 'cfg-name': 'BasketballDrillText.cfg'},
                'ArenaOfValor': {'file-name': 'ArenaOfValor_1920x1080_60_8bit_420', 'cfg-name': 'ArenaOfValor.cfg'},
                'SlideEditing': {'file-name': 'SlideEditing_1280x720_30', 'cfg-name': 'SlideEditing.cfg'},
                'SlideShow': {'file-name': 'SlideShow_1280x720_20', 'cfg-name': 'SlideShow.cfg'}
                
                }

config_files_sdr = ['ArenaOfValor.cfg',
                    'BasketballDrill.cfg',
                    'BasketballDrillText.cfg',
                    'BasketballDrive.cfg',
                    'BasketballPass.cfg',
                    'BlowingBubbles.cfg',
                    'BQMall.cfg',
                    'BQSquare.cfg',
                    'BQTerrace.cfg',
                    'Bubbles_RGB_16bit.cfg',
                    'Cactus.cfg',
                    'CADWaveform_444.cfg',
                    'CADWaveform_GBR.cfg',
                    'CADWaveform_RGB.cfg',
                    'Campfire.cfg',
                    'Cardiac_400_12bit.cfg',
                    'CatRobot.cfg',
                    'ChinaSpeed.cfg',
                    'CrowdRun_RGB_16bit.cfg',
                    'DaylightRoad2.cfg',
                    'Doc_444.cfg',
                    'Doc_RGB.cfg',
                    'DucksTakeOff_RGB_16bit.cfg',
                    'EBURainFruits_RGB_10bit+2MSB.cfg',
                    'EBURainFruits_RGB_10bit+4MSB.cfg',
                    'EBURainFruits_RGB_10bit+6MSB.cfg',
                    'FoodMarket4.cfg',
                    'FourPeople.cfg',
                    'FruitStall_RGB_16bit.cfg',
                    'Head_400_16bit.cfg',
                    'InToTree_RGB_16bit.cfg',
                    'Johnny.cfg',
                    'Kimono.cfg',
                    'Kimono_RGB_10bit+2MSB.cfg',
                    'Kimono_RGB_10bit+4MSB.cfg',
                    'Kimono_RGB_10bit+6MSB.cfg',
                    'KristenAndSara.cfg',
                    'LongRunShort_400_12bit.cfg',
                    'Map_GBR.cfg',
                    'MarketPlace.cfg',
                    'NebutaFestival_10bit.cfg',
                    'OldTownCross_RGB_16bit.cfg',
                    'ParkJoy_RGB_16bit.cfg',
                    'ParkRunning3.cfg',
                    'ParkScene.cfg',
                    'PartyScene.cfg',
                    'PCBLayout_444.cfg',
                    'PCBLayout_GBR.cfg',
                    'PCBLayout_RGB.cfg',
                    'PeopleOnStreet.cfg',
                    'ppt_doc_xls_444.cfg',
                    'ppt_doc_xls_GBR.cfg',
                    'ppt_doc_xls_RGB.cfg',
                    'Programming_GBR.cfg',
                    'RaceHorses.cfg',
                    'RaceHorsesC.cfg',
                    'RitualDance.cfg',
                    'SlideEditing.cfg',
                    'SlideShow.cfg',
                    'SocialNetworkMap_444.cfg',
                    'SocialNetworkMap_RGB.cfg',
                    'SteamLocomotiveTrain_10bit.cfg',
                    'Tango2.cfg',
                    'Traffic.cfg',
                    'TwistTunnel_444.cfg',
                    'TwistTunnel_GBR.cfg',
                    'TwistTunnel_RGB.cfg',
                    'VenueVu_GBR.cfg',
                    'VideoConferencingDocSharing_444.cfg',
                    'VideoConferencingDocSharing_GBR.cfg',
                    'VideoConferencingDocSharing_RGB.cfg',
                    'Vidyo1.cfg',
                    'Vidyo3.cfg',
                    'Vidyo4.cfg',
                    'WebBrowsing_GBR.cfg',
                    'Web_444.cfg',
                    'Web_RGB.cfg',
                    'WordEditing_444.cfg',
                    'WordEditing_GBR.cfg',
                    'WordEditing_RGB.cfg']


#------------------------------------------------------------------------------

def read_seq_config(config_path):
    with open(config_path) as f:
        lines = f.readlines()
    
    # InputFile                     : Campfire_3840x2160_30fps_10bit_420_bt709_videoRange.yuv
    # InputBitDepth                 : 10          # Input bitdepth
    # InputChromaFormat             : 420         # Ratio of luminance to chrominance samples
    # FrameRate                     : 30          # Frame Rate per second
    # FrameSkip                     : 0           # Number of frames to be skipped in input
    # SourceWidth                   : 3840        # Input  frame width
    # SourceHeight                  : 2160        # Input  frame height
    # FramesToBeEncoded             : 300         # Number of frames to be code
    # Level                         : 5.1
    
    file_name, nbit, chroma_fmt, frate, fskip, fwidth, fheight, nframes, level = '', 8, 420, 30, 0, 0, 0, 0, 0

    for line in lines:
        if line[0] == '#' or line == '\n':
            continue
        key, value = line.split(':')
        if key.strip() == 'InputFile':
            file_name = value.strip()
        elif key.strip() == 'InputBitDepth':
            nbit = int(value.strip().split('#')[0])
        elif key.strip() == 'InputChromaFormat':
            chroma_fmt = int(value.strip().split('#')[0])
        elif key.strip() == 'FrameRate':
           frate = int(value.strip().split('#')[0])
        elif key.strip() == 'FrameSkip':
           fskip = int(value.strip().split('#')[0])
        elif key.strip() == 'SourceWidth':
           fwidth = int(value.strip().split('#')[0])
        elif key.strip() == 'SourceHeight':
           fheight = int(value.strip().split('#')[0])
        elif key.strip() == 'FramesToBeEncoded':
           nframes = int(value.strip().split('#')[0])
        elif key.strip() == 'Level':
           level = float(value.strip())  
        else:
            print('invalid config parameter')
        
    return file_name, nbit, chroma_fmt, frate, fskip, fwidth, fheight, nframes, level
#------------------------------------------------------------------------------