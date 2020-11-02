let model;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
var net;
var webcam;

async function app() {
  net = await mobilenet.load();
  //obtenemos datos del webcam
  webcam = await tf.data.webcam(webcamElement);
  //y los vamos procesando
  while (true) {
    const img = await webcam.capture();
    const activation = net.infer(img, 'conv_preds');
    var result;
    try {
      result = await classifier.predictClass(activation);
    } catch (error) {
      result = {};
    }

    const classes = [
      'Untrained',
      'Hi',
      'Good morning',
      'My name is',
      'Daniel',
      'Nice to meet you!',
    ];

    try {
      document.getElementById('console').innerText = `
    ${classes[result.label]}\n`;
      console.log(`probability: ${result.confidences[result.label]}`);
    } catch (error) {
      document.getElementById('console').innerText =
        'Untrained, follow the instructions';
    }

    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

//add example
async function addExample(classId) {
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);
  //liberamos el tensor
  img.dispose();
}

//Modal functions
const $modal = document.getElementById('modal');
const $overlay = document.getElementById('overlay');
const $hideModal = document.getElementById('hide-modal');

const showModal = ($element) => {
  $overlay.classList.add('active');
  $modal.style.animation = 'modalIn .8s forwards';
};

const hideModal = () => {
  $overlay.classList.remove('active');
  $modal.style.animation = 'modalOut .8s forwards';
};

const startApp = () => {
  document.getElementById('handImg').style = 'display:none;';
  showModal();
  app();
};
