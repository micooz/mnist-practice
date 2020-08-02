import { DataSet, DataValue } from './types';

// information entropy
export function entropy(dataset: DataSet) {
  const kMap = new Map<DataValue, number>();

  for (let i = 0; i < dataset.length; i++) {
    const { label } = dataset[i];
    const count = kMap.get(label) | 0;
    if (!count) {
      kMap.set(label, 0);
    }
    kMap.set(label, count + 1);
  }

  let sum = 0;

  for (const count of kMap.values()) {
    const pk = count / dataset.length;
    sum += pk * Math.log2(pk);
  }

  return -sum;
}

// information gain
export function gain(dataset: DataSet, featureName: string, prevEnt: number) {
  const dsMap = divide(dataset, featureName);

  const ent = [...dsMap.values()].reduce((acc, next) => {
    const weight = next.length / dataset.length
    acc += weight * entropy(next);
    return acc;
  }, 0);

  return {
    value: prevEnt - ent,
    dsMap,
  };
}

function divide(dataset: DataSet, featureName: string) {
  const dsMap = new Map<DataValue, DataSet>();

  for (const item of dataset) {
    const { features } = item;
    const featureValue = features[featureName];

    const ds = dsMap.get(featureValue);
    if (!ds) {
      dsMap.set(featureValue, [item]);
    } else {
      ds.push(item);
    }
  }

  return dsMap;
}
