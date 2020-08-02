import { DataSet, DataValue, FeatureMeta } from './types';

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

// information gain (ID3 & C4.5)
export function gain(dataset: DataSet, featureMeta: FeatureMeta, prevEnt: number) {
  const dsMap = divide(dataset, featureMeta);

  const [ent, iv] = [...dsMap.values()].reduce((acc, next) => {
    const weight = next.length / dataset.length
    acc[0] += weight * entropy(next); // ID3
    acc[1] -= weight * Math.log2(weight); // C4.5
    return acc;
  }, [0, 0]);

  const gainValue = prevEnt - ent;

  return {
    gain: gainValue,
    gainRatio: gainValue / iv, // intrinsic value
    dsMap,
  };
}

function divide(dataset: DataSet, featureMeta: FeatureMeta) {
  const dsMap = new Map<DataValue, DataSet>();

  for (const item of dataset) {
    const { features } = item;
    const featureValue = features[featureMeta.name];

    const ds = dsMap.get(featureValue);
    if (!ds) {
      dsMap.set(featureValue, [item]);
    } else {
      ds.push(item);
    }
  }

  return dsMap;
}
