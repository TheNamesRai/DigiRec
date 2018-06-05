(function() {
    function createCanvas(parent, width, height) {
        var canvas = {};
        canvas.node = document.createElement('canvas');
        canvas.node.setAttribute("id", "can");
        canvas.context = canvas.node.getContext('2d');
        canvas.node.width = width || 100;
        canvas.node.height = height || 100;
        parent.appendChild(canvas.node);
        return canvas;
    }

    function init(container, width, height, fillColor) {
        var canvas = createCanvas(container, width, height);
        var ctx = canvas.context;
        ctx.fillCircle = function(x, y, radius, fillColor) {
            this.fillStyle = fillColor;
            this.beginPath();
            this.moveTo(x, y);
            this.arc(x, y, radius, 0, Math.PI * 2, false);
            this.fill();
        };
        ctx.clearTo = function(fillColor) {
            ctx.fillStyle = fillColor;
            ctx.fillRect(0, 0, width, height);
        };
        ctx.clearTo(fillColor || "#ddd");

        canvas.node.onmousemove = function(e) {
            if (!canvas.isDrawing) {
               return;
            }
            var x = e.pageX - this.offsetLeft;
            var y = e.pageY - this.offsetTop;
            var radius = 10; // or whatever
            var fillColor = '#ff0000';
            ctx.fillCircle(x, y, radius, fillColor);
        };
        canvas.node.onmousedown = function(e) {
            canvas.isDrawing = true;

            var x = e.pageX - this.offsetLeft;
            var y = e.pageY - this.offsetTop;
            var radius = 10; // or whatever
            var fillColor = '#ff0000';
            ctx.fillCircle(x, y, radius, fillColor);
        };
        canvas.node.onmouseup = function(e) {
            canvas.isDrawing = false;
        };
    }

    var container = document.getElementById('canvas');
    init(container, 280, 280, '#ddd');

})();

function getData() {
    var c = document.getElementById("can");
    var ctx = c.getContext("2d");
    var imgData = ctx.getImageData(0, 0, c.width, c.height);
    console.log(imgData.data);
}

function toGrayScale(){
    var c = document.getElementById("can");
    var context = c.getContext("2d");
    var imageData = context.getImageData(0,0,c.width,c.height);
    var data = imageData.data;
    console.log(data);
    for(var i=0; i<data.length; i+=4){
        var avg = (data[i]+data[i+1]+data[i+2])/3;
        data[i] = avg;
        data[i+1] = avg;
        data[i+2] = avg;
    }
    console.log(data);
    imageData.data = data;
    var grayCanvas = document.getElementById("graycanvas");
    var grayContext = graycanvas.getContext("2d");   
    grayContext.putImageData(imageData, 0, 0);
}

function clearCanvas(){
    var c = document.getElementById("can");
    var ctx = c.getContext("2d");
    ctx.clearRect(0,0, c.width, c.height);

    var c = document.getElementById("graycanvas");
    var ctx = c.getContext("2d");
    ctx.clearRect(0,0, c.width, c.height);

    var c = document.getElementById("compressed");
    var ctx = c.getContext("2d");
    ctx.clearRect(0,0, c.width, c.height);
}

function compress(){
    var c = document.getElementById("graycanvas");
    var ctx = c.getContext("2d");
    var imageData = ctx.getImageData(0,0, c.width, c.height);
    var data = imageData.data;
    // var 
    for(var i=0; i<data.length; i+=10){
        // for()
    }
    imageData.data = data;
    var compCanvas = document.getElementById("compressed");
    var compCtx = compCanvas.getContext("2d");
    compCtx.putImageData(imageData, 0, 0);
}