from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


annType = 'keypoints'

cocoGt=COCO('pose_label_validate.json')
cocoDt=cocoGt.loadRes('hg_result.json')

imgIds=sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
