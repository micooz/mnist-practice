import * as fs from 'fs';
import * as Bmp from 'binary-bmp';

export function saveImage(
  name: string,
  buf: Buffer,
  width: number,
  height: number,
) {
  const bmp = new Bmp(8, {
    data: buf,
    width,
    height,
  });

  const filename = `tmp/${name}.bmp`;

  console.log(`write ${filename}`);

  fs.writeFileSync(filename, bmp.getBuffer());
}
