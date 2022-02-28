
import csv
import numpy

def load_csv_file(filename: str, header_elements=None):

    with open(filename, newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_header = None
        csv_idx = None

        images = []
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                csv_header = row

                if header_elements is not None:
                    new_csv_header = ["ID"]
                    for landmark in header_elements:
                        new_csv_header.append(landmark)
                csv_idx = []

                if header_elements is not None:
                    for idx_2, element_header in enumerate(csv_header):
                        if element_header in new_csv_header:
                            csv_idx.append(idx_2)

                    csv_header = new_csv_header
            else:
                if header_elements is not None:
                    filtered_row = []
                    for idx in csv_idx:
                            filtered_row.append(row[idx])

                            images.append(filtered_row)
                else:
                    images.append(row)

        return images, csv_header


from decimal import Decimal
from visdom import Visdom
viz = Visdom(port=8850)

filename='../BRATS2020_npz/database.csv'
with open(filename, newline='', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)
    reader = csv.DictReader(csvfile)
    i=0
    for row in reader:
        dec = Decimal(row['tumor_amount'])
        print(dec, type(dec))
        if row['phase']=='train' and dec>0.05:
            print(row['tumor_amount'])
            print(row['fp'], row['tumor_amount'])
            i+=1
            print('i',i)
            img=numpy.load('../BRATS2020_npz/'+row['fp'])

            print('img', img.files)
            a=img['img']
            print('a', a.shape)
            # viz.image(a[0, ...])
            # viz.image(a[1, ...])
            # viz.image(a[2, ...])
            # viz.image(a[3, ...])
            # viz.image(a[4, ...])
            numpy.save('../bratshalf/train/krank/'+row['fp'].split('/')[-1], a)