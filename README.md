Background removal for Unacademy
-------------
<ol>
We have 3 ways to remove the background
<li> Browser side tensorflow.js </li>
<li> Server side opencv for plain background removal </li>
<li> Server side trained model for complex background removal </li>
</ol>


Run the tensorflow.js model
-------------
Try the browser side tensorflow.js live on http://bit.ly/unaback

The description and install of pretrianed tensorflow.js model 
-------------
Checkout https://github.com/4g/unacademy/tree/master/tfjs-models



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


