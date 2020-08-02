import { DataSet, DataValue, Features } from './types';
import * as utils from './utils';

class Node {
  featureName: string;
  branches: Branch[];
  leaf?: true;
  label?: DataValue;
}

class Branch {
  featureValue: DataValue;
  node: Node;
}

export function createTree(dataset: DataSet, featureNames: string[]) {
  if (featureNames.length === 0) {
    return createLeafNode(getMaxLabel(dataset));
  }

  // all data labels are the same
  const labelSet = new Set(dataset.map(item => item.label));
  if (labelSet.size === 1) {
    return createLeafNode(labelSet.values().next().value);
  }

  const node = new Node();

  const ent = utils.entropy(dataset);
  // console.log(ent);

  let maxGain = 0;
  let resultFeatureName = '';
  let resultDsMap: Map<DataValue, DataSet> = new Map();

  // find best divide point
  featureNames.forEach(name => {
    const { value, dsMap } = utils.gain(dataset, name, ent);

    if (value > maxGain) {
      maxGain = value;
      resultFeatureName = name;
      resultDsMap = dsMap;
    }
  });

  node.featureName = resultFeatureName;
  node.branches = [...resultDsMap.entries()].map(([featureValue, ds]) => {
    const newFeatureNames =
      featureNames.filter(name => name !== resultFeatureName);

    const branch = new Branch();
    branch.featureValue = featureValue;
    branch.node = createTree(ds, newFeatureNames);

    return branch;
  });

  // TODO: complement feature values of featureName
  // if (node.featureName === '色泽') {
  //   const branch = new Branch();
  //   branch.featureValue = '浅白';
  //   branch.node = createLeafNode(getMaxLabel(dataset));
  //   node.branches.push(branch);
  // }

  return node;
}

export function predict(tree: Node, features: Features): DataValue {
  const { featureName, branches, leaf, label } = tree;

  if (leaf) {
    return label;
  }

  const featureValue = features[featureName];
  if (!featureValue) {
    return null;
  }

  for (const branch of branches) {
    if (branch.featureValue === featureValue) {
      return predict(branch.node, features);
    }
  }

  return null;
}

function createLeafNode(label: DataValue) {
  const node = new Node();
  node.leaf = true;
  node.label = label;
  return node;
}

function getMaxLabel(dataset: DataSet) {
  const map = new Map<DataValue, number>();

  dataset.forEach(({ label }) => {
    const count = map.get(label);
    if (!count) {
      map.set(label, 0);
    } else {
      map.set(label, count + 1);
    }
  });

  const result = [...map.entries()].sort((a, b) => {
    return b[1] - a[1];
  });

  return result[0][0];
}
