class SignTool {
    constructor() {
        this.initVars()
        this.initEvents()
    }

    initVars() {
        this.canvas = $('#canvas')[0]
        this.ctx = this.canvas.getContext("2d")
        this.isMouseClicked = false
        this.isMouseInCanvas = false
        this.prevX = 0
        this.currX = 0
        this.prevY = 0
        this.currY = 0
    }

    initEvents() {
        $('#canvas').on("mousemove", (e) => this.onMouseMove(e))
        $('#canvas').on("mousedown", (e) => this.onMouseDown(e))
        $('#canvas').on("mouseup", () => this.onMouseUp())
        $('#canvas').on("mouseout", () => this.onMouseOut())
        $('#canvas').on("mouseenter", (e) => this.onMouseEnter(e))
    }

    onMouseDown(e) {
    	this.isMouseClicked = true
        this.updateCurrentPosition(e)
    }

    onMouseUp() {
    	this.isMouseClicked = false
    }

    onMouseEnter(e) {
    	this.isMouseInCanvas = true
        this.updateCurrentPosition(e)
    }

    onMouseOut() {
    	this.isMouseInCanvas = false
    }

    onMouseMove(e) {
        if (this.isMouseClicked && this.isMouseInCanvas) {
      	    this.updateCurrentPosition(e)
            this.draw()
        }
    }

    updateCurrentPosition(e) {
        this.prevX = this.currX
        this.prevY = this.currY
        this.currX = e.clientX - this.canvas.offsetLeft
        this.currY = e.clientY - this.canvas.offsetTop
    }

    draw() {
        this.ctx.beginPath()
        this.ctx.moveTo(this.prevX, this.prevY)
        this.ctx.lineTo(this.currX, this.currY)
        this.ctx.strokeStyle = "black"
        this.ctx.lineWidth = 20
        this.ctx.stroke()
        this.ctx.closePath()
    }
}

var canvas = new SignTool()

$(document).ready(function(){
    $('#submit').click(function(){
        var dataURL = $('#canvas')[0].toDataURL();
        var data = JSON.stringify({ 
            contentType: 'application/json',
            image: dataURL
        })
        //console.log(data)
        $.ajax({
          type: "POST",
          headers: {                                  
            'Accept': 'application/json',
            'Content-Type': 'application/json' 
            },
          url: "/api/upload",
          data,
          success: function(response) {
            console.log('saved')
            console.log(response)
          }
        })
    });
});
