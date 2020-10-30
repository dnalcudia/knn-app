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

const saveKnn = async () => {
  //obtenemos el dataset actual del clasificador (labels y vectores)
  let strClassifier = JSON.stringify(
    Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [
      label,
      Array.from(data.dataSync()),
      data.shape,
    ])
  );
  const storageKey = 'knnClassifier';
  //lo almacenamos en el localStorage
  localStorage.setItem(storageKey, strClassifier);
};

const loadKnn = async () => {
  const storageKey = 'knnClassifier';
  let datasetJson = localStorage.getItem(storageKey);
  classifier.setClassifierDataset(
    Object.fromEntries(
      JSON.parse(datasetJson).map(([label, data, shape]) => [
        label,
        tf.tensor(data, shape),
      ])
    )
  );
};

const startApp = () => {
  const handImg = document.getElementById('handImg');
  handImg.style = 'display:none;';
  app();
};
