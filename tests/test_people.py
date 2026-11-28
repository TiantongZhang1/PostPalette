import unittest
from unittest.mock import patch
from io import BytesIO
from PIL import Image
import torch
from scratch.people_networks import PersonPoseNet,ImageClassifier,detections
from scratch.person_train import targets,objective,convert
from people_analysis import analyze_people
from models import ModelError


class PeopleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_two_person_annotations_are_not_collapsed_to_presence(self):
        row={'width':192,'height':192,'people':[{'bbox':[10,10,50,150],'keypoints':[0]*51},
                                              {'bbox':[100,10,60,150],'keypoints':[0]*51}],'crowds':[]}
        t=targets([row],192,[False])
        self.assertEqual(int(t[3].sum()),2)
        model=PersonPoseNet();output=model(torch.zeros(1,3,192,192));loss=objective(output,t)
        self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_decode_returns_two_separate_boxes_and_keypoints(self):
        heat=torch.full((1,1,24,24),-20.);geometry=torch.zeros(1,55,24,24)
        for x in (5,18):
            heat[0,0,12,x]=10.;geometry[0,:2,12,x]=torch.tensor([1.8,2.7])
        result=detections((heat,geometry),192,.5)[0]
        self.assertEqual(len(result),2)
        self.assertEqual(len(result[0]['keypoints']),17)

    def test_crowds_are_ignored_in_heatmap_loss(self):
        row={'width':192,'height':192,'people':[],'crowds':[[0,0,192,192]]}
        t=targets([row],192,[False])
        loss=objective((torch.zeros(1,1,24,24),torch.zeros(1,55,24,24)),t)
        self.assertAlmostEqual(float(loss),0.)

    def test_flip_preserves_people_count_and_flips_boxes(self):
        row={'width':192,'height':192,'people':[{'bbox':[10,20,40,100],'keypoints':[0]*51}],'crowds':[]}
        flipped,_=convert(row,192,True)
        self.assertEqual(flipped[0]['box'],[142,20,182,120])

    def test_people_summary_does_not_assert_relationship(self):
        stream=BytesIO();Image.new('RGB',(192,192)).save(stream,format='PNG')
        record={'size':192,'best_val':.5,'threshold':.3,'step':500,'history':[{'val':{'matched_person_pck_02':.8}}]}
        people=[{'box':[10,10,70,180],'score':.9,'keypoints':[[0,0,0]]*17},
                {'box':[110,10,180,180],'score':.9,'keypoints':[[0,0,0]]*17}]
        with patch('people_analysis.model_status',return_value={'person_pose':True,'action':False,'expression':False}),\
             patch('people_analysis.load_model',return_value=(torch.nn.Identity(),record)),\
             patch('people_analysis.detections',return_value=[people]):
            r=analyze_people(stream.getvalue())
        self.assertEqual(r['count_estimate'],2)
        self.assertIn('是否为合影需要确认',r['summary'])
        self.assertNotIn('朋友',r['summary'])
        self.assertTrue(all(p['expression'] is None for p in r['people']))

    def test_unfinished_detector_does_not_guess_a_count(self):
        stream=BytesIO();Image.new('RGB',(64,64)).save(stream,format='PNG')
        with patch('people_analysis.model_status',return_value={'person_pose':False}):
            with self.assertRaises(ModelError):analyze_people(stream.getvalue())

    def test_padding_candidates_do_not_hide_people_or_trigger_truncation(self):
        stream=BytesIO();Image.new('RGB',(192,64)).save(stream,format='PNG')
        record={'size':192,'best_val':.5,'threshold':.3,'step':500,'history':[{'val':{'matched_person_pck_02':.8}}]}
        def person(box):return {'box':box,'score':.9,'keypoints':[[0,0,0]]*17}
        people=[person([10,0,50,40]) for _ in range(11)]
        people.extend([person([10,64,70,128]),person([100,64,170,128])])
        with patch('people_analysis.model_status',return_value={'person_pose':True,'action':False,'expression':False}),\
             patch('people_analysis.load_model',return_value=(torch.nn.Identity(),record)),\
             patch('people_analysis.detections',return_value=[people]):
            r=analyze_people(stream.getvalue())
        self.assertEqual(r['count_estimate'],2)
        self.assertFalse(r['truncated'])


if __name__=='__main__':unittest.main()
