
// 监听鼠标移动事件
let isDrawing = false;
let isPainting = false;
let isErasing = false;
//上一个鼠标的位置
let lastX = 0;
let lastY = 0;
// 存储绘制线条时的点坐标
let _points = []
let points = [];

// 获取Canvas元素
const canvas = document.getElementById('canvas');
// 获取绘图上下文
const ctx = canvas.getContext('2d');
// 获取 toolbar 元素
const canvasContainer = document.getElementById('canvas');
// 设置Canvas的宽度和高度
canvas.width = canvasContainer.offsetWidth;
canvas.height = canvasContainer.offsetHeight;

  //绘制按钮的函数
function writeCanvas() {
    if (state != 1) {
      displayOfPaintSizeEdit('none')
      //弹出信息框
      const toast = new bootstrap.Toast(document.getElementById('writeToast'))
      toast.show()
  
      canvasContainer.style.cursor = 'url(../static/image/pen2.png) 0 30, pointer';
      state = 1;
      resetBtnOpacity('1');
      writeButton.style.opacity = '0.3'
  
    } else {
      resetState();
    }
  }
  //涂画按钮的函数
  function paintCanvas() {
    if (state != 3) {
      displayOfPaintSizeEdit('flex')
      canvasContainer.style.cursor = 'url(../static/image/paint-brush2.png) 0 30, pointer';
      state = 3;
      resetBtnOpacity('1');
      paintButton.style.opacity = '0.3'
  
    } else {
      resetState();
      displayOfPaintSizeEdit('none')
    }
  }
  //擦除按钮的函数
  function eraseCanvas() {
    if (state != 2) {
      displayOfPaintSizeEdit('none')
  
      canvasContainer.style.cursor = 'url(../static/image/eraser2.png) 15 20, pointer';
      state = 2;
      resetBtnOpacity('1');
      eraseButton.style.opacity = '0.3'
  
    } else {
      resetState();
    }
  }
  //清除按钮的函数
  function clearCanvas() {
    displayOfPaintSizeEdit('none')
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // 清空点坐标数组
    points = [];
    resetState();
  }
  //恢复绘制状态为无
  function resetState() {
    displayOfPaintSizeEdit('none')
    canvasContainer.style.cursor = 'auto';
    state = 0;
    resetBtnOpacity('1');
  }
  //canvas鼠标按下监听
  canvas.addEventListener('mousedown', (e) => {
    if (state === 1) {
      isDrawing = true;
      console.log(e.clientX, canvas.offsetLeft, e.clientY, canvas.offsetTop);
      [lastX, lastY] = [e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop];
    }
    else if (state === 2) {
      isErasing = true;
      erase(e);
    }
    else if (state === 3) {
      isPainting = true;
      [lastX, lastY] = [e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop];
    }
  });
  //canvas鼠标按住监听
  canvas.addEventListener('mousemove', (e) => {
    if (state === 1) {
      if (!isDrawing) return;
      //现在的鼠标位置
      const currentX = e.clientX - canvas.offsetLeft;
      const currentY = e.clientY - canvas.offsetTop;
      // 添加点坐标到数组
      _points.push({ x: currentX, y: currentY });
      // 绘制线条
      drawLine(lastX, lastY, currentX, currentY);
      [lastX, lastY] = [currentX, currentY];
    } else if (state === 3) {
      if (!isPainting) return;
      //现在的鼠标位置
      const currentX = e.clientX - canvas.offsetLeft;
      const currentY = e.clientY - canvas.offsetTop;
      //绘制区域
      paintLine(lastX, lastY, currentX, currentY);
      [lastX, lastY] = [currentX, currentY];
    }
    else if (state === 2) {
      if (!isErasing) return;
      erase(e);
    }
  });
  //canvas鼠标松开监听
  canvas.addEventListener('mouseup', () => {
    if (state === 1) {
      isDrawing = false;
      if (_points.length > 0) {
        points.push(_points);
        _points = [];
      }
  
    } else if (state === 3) {
      isPainting = false;
    }
    else if (state === 2) {
      isErasing = false;
    }
    console.log(points);
  });
  // 绘制线条
  function drawLine(startX, startY, endX, endY) {
    ctx.globalCompositeOperation = 'source-over';
  
    // 设置线段的虚线样式
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.lineWidth = 3; // 设置线条宽度
    ctx.lineJoin = 'round'; // 设置线条连接方式为圆角
    ctx.lineCap = 'round'; // 设置线条末端样式为圆角
    ctx.stroke();
  }
  // 绘制区域
  function paintLine(startX, startY, endX, endY) {
    ctx.globalCompositeOperation = 'source-over';
  
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.lineWidth = paintArea; // 设置线条宽度
    ctx.lineJoin = 'round'; // 设置线条连接方式为圆角
    ctx.lineCap = 'round'; // 设置线条末端样式为圆角
    ctx.stroke();
  }
  // 擦除绘图
  function erase(event) {
    const x = event.clientX - canvas.offsetLeft;
    const y = event.clientY - canvas.offsetTop;
  
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    //橡皮配置
    ctx.arc(x, y, 20, 0, 2 * Math.PI);
    ctx.fill();
    // 遍历点坐标数组，并删除与擦除位置相近的点
    for (let i = points.length - 1; i >= 0; i--) {
      for (let j = points[i].length - 1; j >= 0; j--) {
        const point = points[i][j];
        const dx = point.x - x;
        const dy = point.y - y;
        const distance = Math.sqrt(dx * dx + dy * dy);
  
        if (distance < 20) {
          points[i].splice(j, 1);
          if (points[i].length == 0) {
            points.pop(points[i]);
          }
        }
      }
    }
  }
  // 导出 Canvas 内容为 PNG 图像
function exportCanvas() {
    // 创建请求体对象
    const canvasHeight = canvas.height;
    const canvasWidth = canvas.width;
    // 将画布内容转换为 PNG 数据 URL
    var base64Data = canvas.toDataURL('image/png').split(',')[1];
    const request = {
      points: points,
      canvasHeight: canvasHeight,
      canvasWidth: canvasWidth,
      image: base64Data
    };
    // 发送 POST 请求到后端
    fetch('/uploadMask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    })
      .then(response => response.json()) // 解析响应为 JSON
      .then(data => {
        // 处理响应数据
        console.log(data);
        if (data.status === 0) {
          // 创建新的Image对象
          const image = new Image();
          // 设置图像源为解码后的图像数据
          image.src = 'data:image/png;base64,' + data.canvas;
          // 添加 CSS 类
          image.classList.add('mask');
  
          // 将图像添加到页面中的某个元素
          var imageElements = document.getElementById('editing-mask-btn').querySelectorAll('.mask');
          imageElements.forEach(function(element) {
              document.getElementById('editing-mask-btn').removeChild(element);
          });
          document.getElementById('editing-mask-btn').appendChild(image);
          mask = data.canvas;
  
          //判断有没有别的mask
          var maskParent = document.getElementById('upload-mask-btn');
          var masks = maskParent.querySelectorAll('.mask');
          masks.forEach(function(mask) {
            maskParent.removeChild(mask);
          });
        } else {
          //弹出信息框
          const toast = new bootstrap.Toast(document.getElementById('maskNullToast'))
          toast.show()
        }
  
        //一些ui
        maskButton.classList.remove('border-green');
        displayOfEditingText('none')
        maskState = 0;
        displayOfTool('none')
        clearCanvas();
  
      })
      .catch(error => {
        // 处理错误
        console.error(error);
      });
  
  }