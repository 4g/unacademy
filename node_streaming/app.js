const tf = require('@tensorflow/tfjs')
// Load the binding (CPU computation)
// require('@tensorflow/tfjs-node')
// // Or load the binding (GPU computation)
// require('@tensorflow/tfjs-node-gpu')
require('@tensorflow/tfjs-node')
bodyPix = require('@tensorflow-models/body-pix')
var Promise = require('bluebird');
var express = require('express');
var app = express()
bodyParser = require('body-parser'),
errorHandler = require('errorhandler'),
methodOverride = require('method-override')
const fs = require('fs')
const path = require('path')

const {
    createCanvas, Image, ImageData
} = require('canvas')
const imageScaleFactor = 0.5;
const outputStride = 16;
const flipHorizontal = false;

app.use(express.static(path.join(__dirname, 'public')))
app.use(methodOverride());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));

app.use(function (req, res, next) {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE, OPTIONS');
    res.header(
        'Access-Control-Allow-Headers',
        'X-Requested-With, Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Methods, Authorization'
    );
    next();
});

app.get('/', function(req, res) {
  res.sendFile(path.join(__dirname + '/index.htm'))
})

app.get('/healthCheck', function(req, res){
    return res.json({code: 200, status: 'green'})
})


var loadModel = async function () {
    // let modelLoad = bodyPix.load()
    modelLoad = await bodyPix.load()
    return modelLoad
}

function sleep(ms){
    return new Promise(resolve=>{
        setTimeout(resolve,ms)
    })
}

var process = async function(imgPath) {
        const img = new Image();
        img.src = imgPath
        const canvas = createCanvas(img.width, img.height);
        const ctx = canvas.getContext('2d');
        await ctx.drawImage(img, 0, 0);
        const input = tf.browser.fromPixels(canvas);
        let net = await loadModel()
        const outputStride = 16;
        const segmentationThreshold = 0.5;
        const segmentation =  await net.estimatePersonSegmentation(input, outputStride, segmentationThreshold);
        maskBackground = false
        var width = segmentation.width, height = segmentation.height, data = segmentation.data;
        var bytes = new Uint8ClampedArray(width * height * 4);
        for (var i = 0; i < height * width; ++i) {
            var shouldMask = maskBackground ? 1 - data[i] : data[i];
            var alpha = shouldMask * 255;
            var j = i * 4;
            bytes[j + 0] = 0;
            bytes[j + 1] = 0;
            bytes[j + 2] = 0;
            bytes[j + 3] = Math.round(alpha);
        }
        mask =  new ImageData(bytes, width, height);
        bodyPix.drawMask(canvas, input, mask, 1,0, true);
        canvas.toBlob(function(blob) {
            saveAs(blob, "pretty-image.png");
        });
        return canvas
    
}

app.post('/check', function(req, res) {
    let imageElement = req.body.img 
    return Promise.all([process(imageElement)])
    .then(([r]) => {
        console.log(r)
        const response = {
            img:r,
            code: 200,
        };
        return res.json(response)
    }).
    catch(err => {
        console.log(err)
        return res.status(500).json({
            code: 500,
            error: 'SERVER ERROR',
            debug: err
        });
    });
})

app.get('/video', async function(req, res) {
    const path = 'assets/sample.mp4'
    const stat = fs.statSync(path)
    const fileSize = stat.size
    const range = req.headers.range
    if (range) {
      const parts = range.replace(/bytes=/, "").split("-")
      const start = parseInt(parts[0], 10)
      const end = parts[1] 
        ? parseInt(parts[1], 10)
        : fileSize-1
      const chunksize = (end-start)+1
      const file = fs.createReadStream(path, {start, end})
      bodyPix = process(file.path)
      await process(file.path)
        const head = {
            'Content-Range': `bytes ${start}-${end}/${fileSize}`,
            'Accept-Ranges': 'bytes',
            'Content-Length': chunksize,
            'Content-Type': 'video/mp4',
            }
            res.writeHead(206, head);
        file.pipe(res);
    } else {
      const head = {
        'Content-Length': fileSize,
        'Content-Type': 'video/mp4',
      }
      res.writeHead(200, head)
    fs.createReadStream(path).pipe(res)
    }
  });

// logger.info('Simple static server showing %s listening at http://%s:%s', publicDir, hostname, port);
app.listen(8005, 'localhost');
