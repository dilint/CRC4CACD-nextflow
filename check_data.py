import json, glob, os, argparse, datetime, sys
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check data')
    # parser.add_argument('--work-path', type=str, default='.', help='work path of workflow')
    parser.add_argument('--input-dir', type=str, default='test_data', help='input directory')
    parser.add_argument('--log-path', type=str, default='error.txt', help='log file')
    parser.add_argument('--output-path', type=str, default='test.json', help='json file for test data')
    args = parser.parse_args()
    input_dir = args.input_dir
    log_path = args.log_path
    output_path = args.output_path
    
    categories_item = [
            {
                "id": 1,
                "supercategory": "none",
                "name": "ascus"
            },
            {
                "id": 2,
                "supercategory": "none",
                "name": "asch"
            },
            {
                "id": 3,
                "supercategory": "none",
                "name": "lsil"
            },
            {
                "id": 4,
                "supercategory": "none",
                "name": "hsil"
            },
            {
                "id": 5,
                "supercategory": "none",
                "name": "scc"
            },
            {
                "id": 6,
                "supercategory": "none",
                "name": "agc"
            },
            {
                "id": 7,
                "supercategory": "none",
                "name": "trichomonas"
            },
            {
                "id": 8,
                "supercategory": "none",
                "name": "candida"
            },
            {
                "id": 9,
                "supercategory": "none",
                "name": "flora"
            },
            {
                "id": 10,
                "supercategory": "none",
                "name": "herps"
            },
            {
                "id": 11,
                "supercategory": "none",
                "name": "actinomyces"
            }
    ]

    img_paths = []
    for file_name in os.listdir(input_dir):
        _, extension = os.path.splitext(file_name)
        if extension not in ['.bmp', '.jpg', '.jpeg', '.png']:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            open(log_path, 'a').write(f'{current_time} - Please check ***{file_name}***, only support bmp, jpg, jpeg, png format!\n')
        else:
            img_paths.append(os.path.join(input_dir, file_name))

    # 打印图片路径
    id = 0
    images_item = []
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        image = Image.open(img_path)
        width, height = image.size
        img_dict = {
            'file_name': img_name,
            'id': id,
            'width': width,
            'height': height
        }
        id += 1
        images_item.append(img_dict)
    output_data = {
        'categories': categories_item,
        'images': images_item,
        'type': 'instances',
        'annotations': []
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
