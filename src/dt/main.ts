import * as assert from 'assert';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { DataSet, DataItem, FeatureMeta } from './types';
import * as tree from './tree';

const MODEL_PATH = 'model/dt/model.json';

async function getDataset(): Promise<{ dataset: DataSet; featureMetas: FeatureMeta[]; }> {
  const csvDataset = tf.data.csv('file://src/dt/watermelon2.0.csv', {
    columnConfigs: {
      好瓜: {
        isLabel: true,
      },
    },
  });

  const rawData = await csvDataset.toArray();

  // @ts-ignore
  const dataset = rawData.map(({ xs, ys }) => {
    // skip first column
    delete xs['编号'];

    return {
      features: xs,
      label: Object.values(ys)[0],
    } as DataItem;
  });

  return {
    dataset,
    featureMetas: [
      { name: '色泽', type: 'discrete', enums: ['青绿', '乌黑', '浅白'] },
      { name: '根蒂', type: 'discrete', enums: ['蜷缩', '稍蜷', '硬挺'] },
      { name: '敲声', type: 'discrete', enums: ['浊响', '沉闷', '清脆'] },
      { name: '纹理', type: 'discrete', enums: ['清晰', '稍糊', '模糊'] },
      { name: '脐部', type: 'discrete', enums: ['凹陷', '稍凹', '平坦'] },
      { name: '触感', type: 'discrete', enums: ['硬滑', '软粘'] },
    ],
  };
}

async function train() {
  const { dataset, featureMetas } = await getDataset();
  // console.log(dataset);

  const model = tree.createTree(dataset, featureMetas);

  fs.writeFileSync(
    MODEL_PATH,
    JSON.stringify(model, null, 2),
    'utf8'
  );
}

function test() {
  const model = JSON.parse(
    fs.readFileSync(MODEL_PATH, 'utf8')
  );

  const features = {
    色泽: '浅白',
    根蒂: '稍蜷',
    敲声: '浊响',
    纹理: '清晰',
    脐部: '稍凹',
    触感: '软粘',
  };

  const result = tree.predict(model, features);
  console.log(result);

  assert.equal(result, '是');
}

train().then(test);

// test();
