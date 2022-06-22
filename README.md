# project-akhir-orbit-co-vision
Tracking and Counting Vehicles using yolov5 and DeepSort

1. Sebelumnya, cloning repo : git clone https://github.com/pac3010/project-akhir-orbit-co-vision.git
2. Buka folder, kemudian create virtual environment : python -m venv venv
3. Aktifkan venv (Bisa select interpreter di VsCode)
4. Install requirements : pip install -r requirements.txt

Terdapat dua folder dalam repositori ini.

## Folder yolov5_deepsort_webcam_django
Fungsi folder ini untuk melakukan hosting website Co-Vision guna melakukan _Tracking_ dan _Counting_ dengan webcam.
Langkah-langkah :
1. Masuk ke folder yolov5_deepsort_webcam_django : cd .\yolov5_deepsort_webcam_django
2. Masuk ke folder stream : cd .\stream
3. Jalankan server web django : python manage.py runserver
4. Klik localhost yang muncul.

## Folder yolov5-DeepSort_Using-Input-Video(Local)
Fungsi folder ini untuk melakukan _Tracking_ dan _Counting_ melalui input video.
Langkah-langkah :
1. Masuk ke folder yolov5-DeepSort_Using-Input-Video(Local) : cd .\yolov5-DeepSort_Using-Input-Video(Local)
2. Masuk ke folder Yolov5_DeepSort_Pytorch : cd .\Yolov5_DeepSort_Pytorch
3. Masukkan video yang ingin diproses padacfolder "videos" 
4. Ubah settingan pada file track.py di line ke-294 : parser.add_argument('--source', type=str, default='videos_name.mp4', help='source')  # file/folder, 0 for webcam
5. Jalankan track.py : python track.py
6. Maka akan muncul video yang sedang dilakukan _tracking_ dan _counting_
7. Hasil video tracking akan disimpan pada folder runs>track
