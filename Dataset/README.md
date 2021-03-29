# ASVSpoof 2019 Dataset:

Available on: https://datashare.ed.ac.uk/handle/10283/3336

The ASVSpoof 2019 Dataset provides thousands of Examples of spoofed and bonafide speech. The spoofed Data is obtained using different spoofing techniques.
Each containing designated files for Development, Evaluation and Training. 

Details about the database:
1. Training and development data for the LA scenario are included in 'ASVspoof2019_LA_train'  ' ASVspoof2019_LA_dev'. Training dataset contains audio files with known ground-truth, which can be used to train systems to distinguish genuine and spoofed speech. The development dataset contains audio files with known ground-truth which can be used for the development of spoofing detection algorithms. Likewise, training and development data for the PA scenario are included in 'ASVspoof2019_PA_train'  ' ASVspoof2019_PA_dev'.
2. Evaluation data for LA and PA are available in 'ASVspoof2019_LA_eval'  and 'ASVspoof2019_PA_eval', respectively.
 
3. Dev and eval enrollment data for ASV are available in 'ASVspoof2019_{LA,PA}_dev' and 'ASVspoof2019_{LA,PA}_eval', respectively.
4. Protocol and keys are available in 'ASVspoof2019_LA_{cm,asv}_protocols'  and ASVspoof2019_PA_{cm,asv}_protocols, respectively.
5. Additional README.LA.txt files and README.pA.txt are included in packages. They are the extended version of ASVspoof2019_{LA,PA}_instructions_v1.txt originally used for the challenge to explain the database.
 
Logical Access:
ASVspoof2019_LA_train, ASVspoof2019_LA_dev, and ASVspoof2019_LA_eval contain audio files for training, development, and evaluation. Audio files in these directories is in flac format.
 
Each column of the protocol is formatted as:
 
SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
        	1) SPEAKER_ID:                   	LA_****, a 4-digit speaker ID
        	2) AUDIO_FILE_NAME:       	LA_****, name of the audio file
        	3) SYSTEM_ID:                     	ID of the speech spoofing system (A01 - A19),  or, for  bonafide speech SYSTEM-ID is left blank ('-')
        	4) -:                          	This column is NOT used for LA.
        	5) KEY:                     	'bonafide' for genuine speech, or, 'spoof' for spoofing speech
 
The third column is left blank (-) to make the structure coherent with physical access file list;
 Brief description on LA spoofing systems, where TTS and VC denote text-to-speech and voice-conversion systems:
        	
    	A01       	TTS  	neural waveform model
    	A02       	TTS  	vocoder
    	A03       	TTS  	vocoder
    	A04       	TTS  	waveform concatenation
    	A05       	VC    	vocoder
    	A06       	VC    	spectral filtering
 
Physical Access:
ASVspoof2019_PA_train, ASVspoof2019_PA_dev, and ASVspoof2019_PA_eval contain audio files for training, development, and evaluation.  ASVspoof2019_PA_dev, and ASVspoof2019_PA_eval contain audio files to enroll ASV system. The audio files in the directories are in the flac format.
Each column of the protocol is formatted as:
  
 SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID KEY
 
        	1) SPEAKER_ID:                   	PA_****, a 4-digit speaker ID
        	2) AUDIO_FILE_NAME:     name of the audio file
        	3) ENVIRONMENT_ID: a triplet (S,R,D_s), which take one letter in the set {a,b,c} as categorical value, defined as:
 
        	
 
                                    a               b               c
S:   Room size (square meters)     2-5              5-10 	        10-20
R:   T60 (ms)                     50-200             200-600         600-1000
D_s: Talker-to-ASV distance (cm)  10-50             50-100          100-150

 
 
        	4) ATTACK_ID: a duple (D_a,Q), which take one letter in the set {A,B,C} as categorical value, defined as
 
                                            A               B               C
Z: Attacker-to-talker distance (cm)     10-50            50-100          > 100
Q: Replay device quality               perfect            high 	           low

 
                    	
        	5) KEY: 'bonafide' for genuine speech, or, 'spoof' for spoofing speech
 
 
The whole dataset was split in 2 classes ‘Spoof’ and ‘Bonafide’ using the cm protocols.
