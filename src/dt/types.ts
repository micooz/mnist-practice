export type DataValue = string | number;

export type Features = { [key: string]: DataValue };

export type DataItem = {
  features: Features;
  label: DataValue;
};

export type DataSet = DataItem[];

export type FeatureMeta = {
  name: string;
  type: 'discrete' | 'continuous';
  enums?: DataValue[];
};
