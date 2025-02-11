let processId = '';//记录处理id
let paintArea = 30;//绘画大小

const tool = document.getElementById('tool');
const progressUI = document.getElementById('progress');
const confirmedContainer = document.getElementById('confirmed-container');
const writeButton = document.getElementById('write-btn');
writeButton.addEventListener('click', writeCanvas);
const paintButton = document.getElementById('paint-btn');
paintButton.addEventListener('click', paintCanvas);
const eraseButton = document.getElementById('erase-btn');
eraseButton.addEventListener('click', eraseCanvas);
const clearButton = document.getElementById('clean-btn');
clearButton.addEventListener('click', clearCanvas);
const confirmButton = document.getElementById('confirm-btn');
confirmButton.addEventListener('click', confirmInput);
const exportButton = document.getElementById('export-btn');
exportButton.addEventListener('click', exportCanvas);
const maskButton = document.getElementById('editing-mask-btn');
maskButton.addEventListener('click', triggerMask);
const maskSwitchBtn = document.getElementById('mask-switch-btn');
maskSwitchBtn.addEventListener('click', switchToMask);
const svgSwitchBtn = document.getElementById('svg-switch-btn');
svgSwitchBtn.addEventListener('click', switchToSvg);
const uploadMaskBtn = document.getElementById('upload-mask-btn');
uploadMaskBtn.addEventListener('click', uploadMask);
const uploadSvgBtn = document.getElementById('upload-svg-btn');
uploadSvgBtn.addEventListener('click', uploadSvg);
var imgContainers = document.querySelectorAll('.svg-container');
imgContainers.forEach(function (element) {
  element.addEventListener('click', chooseSvg);
});
var hiddenInputs = document.querySelectorAll('.hidden-input');
hiddenInputs.forEach(function (element) {
  element.style.display = 'none'
});
const editingText = document.getElementById('editing-text')


window.onload = function () {
  displayOfTool('none')
  displayOfEditingText('none')
  displayOfBtnAndProgress('block', 'none')
  displayOfPaintSizeEdit('none')
  displayOfMaskOrSvg('flex', 'none')
  
};

//显示和隐藏调节绘画笔大小的滑块的函数
function displayOfPaintSizeEdit(status) {
  document.getElementById('paint-range-container').style.display = status
}
//改变按钮组的展示状态
function displayOfTool(status) {
  tool.style.display = status
}
//重置按钮透明度
function resetBtnOpacity(opacity) {
  // 获取所有具有editBtn类的元素
  const editButtons = document.getElementsByClassName('editBtn');

  // 遍历每个编辑按钮
  for (let i = 0; i < editButtons.length; i++) {
    const editButton = editButtons[i];

    // 设置按钮的透明度为1
    editButton.style.opacity = opacity;
  }
}

//画笔滑块监听
function changeRange3() {
  // 获取滑块的当前值
  var rangeValue = document.getElementById("paint-range").value;
  paintArea = rangeValue;
}

// 获取进度的函数
function getProgress() {
  // 发送 GET 请求到后端
  fetch('/getProgress', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    },
  })
    .then(response => response.json()) // 解析响应为 JSON
    .then(data => {
      // 处理响应数据
      console.log(data);
      if (data.status === 0) {
        // 状态为0时才读取 SVG 文件并更新进度条
        progressUI.children[0].style.width = data.progress + '%';
        progressUI.children[0].textContent = data.progress + '%';
        //读取svg
        console.log(data.svg);
        // 创建 SVG 元素并设置内容
        const svgElement = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svgElement.innerHTML = data.svg;

        // 找到指定元素并插入 SVG 子元素
        const svgModalBody = document.getElementById('svg-modal-body');
        svgElement.setAttribute('width', '600');
        svgElement.setAttribute('height', '600');
        // 清除 svgModalBody 中的现有内容
        svgModalBody.innerHTML = '';
        svgModalBody.appendChild(svgElement);
    
        
      } 
    })
    .catch(error => {
      // 处理错误
      console.error(error);
    });
}
// 获取GIF
function getGIF() {
  // 发送 GET 请求到后端
  fetch('/getGIF', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ processId: String(processId) })
  })
    .then(response => response.blob()) // 将响应转换为 Blob 对象
    .then(blob => {
      // 创建一个 <a> 元素
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob); // 创建一个临时的 URL
      link.download = processId + '.gif'; // 设置下载的文件名

      // 将 <a> 元素添加到文档中
      document.body.appendChild(link);

      // 模拟点击下载链接
      link.click();

      // 清理临时 URL 和 <a> 元素
      URL.revokeObjectURL(link.href);
      document.body.removeChild(link);
    })
    .catch(error => {
      console.error(error);
    });
}
