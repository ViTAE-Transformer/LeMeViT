import argparse as ag
import json

def get_parser_with_args(metadata_json='change_detection/metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        

    parser.add_argument('--backbone', default=None, type=str, choices=['resnet','swin','vitae','lemevit'], help='type of model')

    parser.add_argument('--dataset', default=None, type=str, choices=['cdd','levir'], help='type of dataset')

    parser.add_argument('--pretrained', default=None, type=str, help='pretrained checkpoint, set None to not load')
    
    parser.add_argument('--exp', default=None, type=str, help='path of saved log')



    return parser, metadata

