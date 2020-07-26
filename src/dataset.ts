import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
// import * as utils from './utils';

function getData(rawImagesFile: string, rawLabelsFile: string, num?: number) {
  let rawImageBuf = fs.readFileSync(rawImagesFile);
  let rawLabelBuf = fs.readFileSync(rawLabelsFile);

  const imagesNum = num || rawImageBuf.readUInt32BE(4);
  const imageWidth = rawImageBuf.readInt32BE(8);
  const imageHeight = rawImageBuf.readInt32BE(12);

  // console.log({ imagesNum, imageWidth, imageHeight });

  // skip first 16 bytes of image buffer
  rawImageBuf = rawImageBuf.slice(16);

  // skip first 8 bytes of label buffer
  rawLabelBuf = rawLabelBuf.slice(8);

  let images: Buffer[] = [];
  let labels: number[] = [];

  for (let i = 0; i < imagesNum; i++) {
    const pos = imageWidth * imageHeight;

    const image = rawImageBuf.slice(pos * i, pos * (i + 1));
    const label = rawLabelBuf.readUInt8(i);

    images.push(image);
    labels.push(label);

    // utils.saveImage(`${i}-${label}`, image, imageWidth, imageHeight);
  }

  return [
    tf.tensor(images, [imagesNum, imageWidth, imageHeight, 1]).div(255).sub(0.5),
    tf.oneHot(labels, 10),
  ];
}

export function getTrainData(num?: number) {
  return getData(
    'mnist/train-images-idx3-ubyte',
    'mnist/train-labels-idx1-ubyte',
    num,
  );
}

export function getTestData(num?: number) {
  return getData(
    'mnist/t10k-images-idx3-ubyte',
    'mnist/t10k-labels-idx1-ubyte',
    num,
  );
}
