import pandas as pd
import typing as t
from operator import methodcaller
from ast import literal_eval

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _eliminate_not_well_tagged,
            _drop_useless,
            _fix_image_column,
            _fix_label
        ]
    )
    df = cleaning_fn(df)
    return df

def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper

def _eliminate_not_well_tagged(df):
    print(df.shape)
    for idx, row in df.iterrows():
        lines = row.label
        for line in lines:
            if(len(line['points'])) != 4:
                df.drop(idx, inplace = True)
                break
        
    print(df.shape)
    return df

def _drop_useless(df):
   return df.drop(['id', 'annotator', 'annotation_id'], axis = 1)

def _fix_image_column(df):

    new_img = df['image'].map(methodcaller('split', "/"))
    df['image'] = ['image/' + x[-1] for x in new_img.values]

    return df

def _fix_label(df):

    def _convert_mask_to_line(lines):
        output = []
        for line in lines:
            points = line['points']
            label = line['polygonlabels'][0]
            x0 = min([points[0][0], points[1][0]]) + abs(points[0][0] - points[1][0])
            x1 = min(points[2][0], points[3][0]) + abs(points[2][0] - points[3][0])

            y0 = min(points[0][1],points[1][1]) + abs(points[0][1] - points[1][1])
            y1 = min(points[2][1],points[3][1]) + abs(points[2][1] - points[3][1])

            if x0 > x1 or y0 > y1:
                x1, x0 = x0, x1
                y1, y0 = y0, y1

            output.append((x0, y0, x1, y1))

        return output
    
    def _get_tag(lines):
        output = []
        for line in lines:
            output.append(line['polygonlabels'][0])

        return output
 
    df['coords'] = df.label.map(_convert_mask_to_line)
    df['tag'] = df.label.map(_get_tag)
    
    return df.drop('label', axis = 1)
