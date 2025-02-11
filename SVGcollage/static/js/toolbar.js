let mask = '';//保存mask
let svg = '';//记录选择的svg
let svgList = []//记录用户上传的svg
let svg_num = 100;//小svg的数量
let iteration = 250;//迭代次数
let state = 0; //0为啥也不干，1为绘画，2为擦除，3为涂刷
let maskState = 0; //0为啥也不干，1为绘制中
//更改上传svg按钮的外观
function changeUploadSvgUI(state){
    if(state===1){//1为上传，0为恢复
        //更改UI
        var i = uploadSvgBtn.querySelector('i');
        i.className = 'bi bi-check-circle';
        i.style.color = 'green';
        i.textContent = svgList.length + ' SVG has been uploaded';
    }else{
        //更改UI
        var i = uploadSvgBtn.querySelector('i');
        i.className = 'bi bi-plus-square-dotted';
        i.style.color = 'black';
        i.textContent ='';
    }
    
  }

  //选择svg时的函数
function chooseSvg(event) {
    if (event.target.getAttribute('svg') != svg) {//如果点击的不是之前的
      resetSvgConfirmIcon('black');
      var targetElement = event.target; // 获取被点击的元素
      var iElement = targetElement.querySelector('i'); // 获取被点击元素下的子元素 i
      // 修改子元素 i 的样式
      iElement.style.color = 'green';
      //设置svg
      svg = targetElement.getAttribute('svg');
      //清空用户输入的
      svgList = [];
      changeUploadSvgUI(0);
    } else {
      resetSvgConfirmIcon('black');
      //设置svg
      svg = '';
    }
  }
  //显示和隐藏ui的函数
function displayOfMaskOrSvg(status1, status2) {
    document.getElementById('mask-edit-container').style.display = status1
    document.getElementById('svg-edit-container').style.display = status2
  }
  //切换面板到mask
  function switchToMask() {
    displayOfMaskOrSvg('flex', 'none')
  }
  //切换面板到svg
  function switchToSvg() {
    displayOfMaskOrSvg('none', 'flex')
  }
  //改变editing的展示状态
  function displayOfEditingText(status) {
    editingText.style.display = status
  }
  
  //小svg列表的icon重置
  function resetSvgConfirmIcon(color) {
    // 获取
    const icons = document.getElementsByClassName('svgConfirmIcon');
    // 遍历
    for (let i = 0; i < icons.length; i++) {
      const icon = icons[i];
      // 设置icon的颜色为黑色
      icon.style.color = color;
    }
  }
  //触发开始mask的绘制
function triggerMask() {
    if (!maskState) {
      displayOfEditingText('block')
      maskButton.classList.add('border-green');
      maskState = 1;
      displayOfTool('flex')
    }
    else {
      maskButton.classList.remove('border-green');
      displayOfEditingText('none')
      maskState = 0;
      displayOfTool('none')
      clearCanvas();
      resetBtnOpacity();
      state = 0;
    }
  
  }
  //滑块1监听
function changeRange1() {
    // 获取滑块的当前值
    var rangeValue = document.getElementById("num-range").value;
    svg_num = rangeValue;
    // 获取原有的字段
    var currentText = document.getElementById("num-range-text").textContent;
    // 拼接新的文本
    var newText = currentText.split(': ')[0] + ': ' + rangeValue;
    // 更新<span>标签的文本
    document.getElementById("num-range-text").textContent = newText;
  }
  //滑块2监听
  function changeRange2() {
    // 获取滑块的当前值
    var rangeValue = document.getElementById("iteration-range").value;
    iteration = rangeValue;
    // 获取原有的字段
    var currentText = document.getElementById("iteration-range-text").textContent;
    // 拼接新的文本
    var newText = currentText.split(': ')[0] + ': ' + rangeValue;
    // 更新<span>标签的文本
    document.getElementById("iteration-range-text").textContent = newText;
  }
  //切换按钮和进度条
function displayOfBtnAndProgress(status1, status2) {
    confirmButton.style.display = status1
    confirmedContainer.style.display = status2
  }
  //本地上传mask
function uploadMask() {
    const fileInput = document.getElementById('upload-mask-input');
    fileInput.click();
    fileInput.onchange = (event) => {
      var fileData = event.target.files[0];
      var pattern = /^image\/(png|jpeg)$/;
      if (typeof (fileData) == "undefined") {
        return;
      }
      if (!pattern.test(fileData.type)) {
        alert("图片格式不正确,请上传jpg或png");
        return;
      }
      var reader = new FileReader();
      reader.readAsDataURL(fileData);
      reader.onload = function () {
        var img = reader.result;
        console.log(img);
  
        base64 = img.split(',')[1];
  
        // 创建新的Image对象
        const image = new Image();
        // 设置图像源为解码后的图像数据
        image.src = 'data:image/png;base64,' + base64;
        // 添加 CSS 类
        image.classList.add('mask');
        // 将图像添加到页面中的某个元素
        var imageElements = document.getElementById('upload-mask-btn').querySelectorAll('.mask');
          imageElements.forEach(function(element) {
              document.getElementById('upload-mask-btn').removeChild(element);
          });
        document.getElementById('upload-mask-btn').appendChild(image);
        mask = base64;
        //判断有没有别的mask
        var maskParent = document.getElementById('editing-mask-btn');
        var masks = maskParent.querySelectorAll('.mask');
        masks.forEach(function(mask) {
          maskParent.removeChild(mask);
        });
      };
    }
  }
  //本地上传Svg
  function uploadSvg() {
    const fileInput = document.getElementById('upload-svg-input');
    fileInput.setAttribute('multiple', 'true');
    fileInput.click();
    fileInput.onchange = (event) => {
      //提前清空列表
      svgList = [];
      //设置svg
      svg = '';
      resetSvgConfirmIcon('black');
      var files = event.target.files;
      var pattern = /^image\/svg\+xml/;
      for (var i = 0; i < files.length; i++) {
        (function (fileData) {
          if (typeof (fileData) == "undefined") {
            return;
          }
          if (!pattern.test(fileData.type)) {
            alert("文件格式不正确，仅支持SVG格式");
            return;
          }
          var reader = new FileReader();
          reader.readAsDataURL(fileData);
          reader.onload = function () {
            var svgData = reader.result;
            svgList.push(svgData.split(',')[1]);
            changeUploadSvgUI(1);
          };
        })(files[i]);
      }
      
    }
  }
  // 开始生成
function confirmInput() {
    if (mask && (svg||svgList.length>0)) {
      // 获取当前时间戳（毫秒）
      processId = Math.floor(Date.now() / 1000);
      console.log(processId)
      const request = {
        mask: mask,
        svg: svg,
        svg_num: svg_num,
        iteration: iteration,
        processId: String(processId),
        svgList:svgList
      };
      // 发送 POST 请求到后端
      fetch('/generateSVG', {
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
  
          if (data.status === 0)
            getGIF();
          else {
            //弹出信息框
            const toast = new bootstrap.Toast(document.getElementById('generateToast'))
            // 获取 toast-body 元素并更改内容
            const toastBody = document.getElementById('generateToast').querySelector('.toast-body');
            toastBody.innerHTML = data.error;
            toast.show()
          }
  
          displayOfBtnAndProgress('block', 'none')
          clearInterval(progressTimer);
        })
        .catch(error => {
          displayOfBtnAndProgress('block', 'none')
          clearInterval(progressTimer);
          // 处理错误
          console.error(error);
        });
      displayOfBtnAndProgress('none', 'flex')
      var progressTimer = setInterval(getProgress, 2000);
    } else {
      //弹出信息框
      const toast = new bootstrap.Toast(document.getElementById('nullToast'))
      toast.show()
    }
  
  }