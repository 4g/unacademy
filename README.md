We have 3 ways to remove the background
1] Browser side tensorflow.js
2] Server side opencv for plain background removal
3] Server side trained model for complex background removal


Try the browser side tensorflow.js live on http://bit.ly/unaback

How to build and run the browser side 
-------------

1] install yarn and npm
2] 'yarn' in tfjs-models/body-pix/
3] 'yarn watch' in tfjs-models/body-pix/demos/
4] Access the url. Allow camera access on the browser



How to build and run the open cv 
-------------
1] git clone this repository
2] Change to opencv code directory
3] Install opencv by pip opencv-python or other methods for different languages
4] Install python flask by pip install flask
5] flask run server.py
6]  flask run client.py
7] Access the browser url of flask

Opencv description
-------------
1] It uses inrange function to pick particular color by hsv model
2] Dilate and erode
3] We find contours of lesser area comapared to the total image area. We then masked them. 

