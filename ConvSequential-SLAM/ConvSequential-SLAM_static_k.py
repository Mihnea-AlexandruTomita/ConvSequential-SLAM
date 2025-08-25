import cv2
import numpy as np
import glob
import os   #Imported here if your image reading methodology uses os.listdirectory sort of implementation.
from Hog_feature.Hog_feature.hog import initialize
from Hog_feature.Hog_feature.hog import extract
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
import csv  
import time
from numba import jit
from scipy import linalg

# ConvSequential-SLAM Parameters
magic_width = 512
magic_height = 512
cell_size = 16      # HOG cell-size
bin_size = 8        # HOG bin size
image_frames = 1    # 1 for grayscale, 3 for RGB
descriptor_depth = bin_size*4*image_frames      # x4 is here for block normalization due to nature of HOG

ET = 0.5   # Entropy threshold, can vary between 0-1
k = 3      # Sequence length

total_Query_Images = 100  # Number of images in the query folder
total_Ref_Images = 100    # Number of images in the reference folder
query_index_offset = 0
ref_index_offset = 0

N_query = total_Query_Images - k
N_ref = total_Ref_Images - k 

total_no_of_regions=int((magic_width/cell_size-1)*(magic_width/cell_size-1))

# Global variables
d1d2dot_matrix=np.zeros([total_no_of_regions,total_no_of_regions],dtype=np.float32)
d1d2matches_maxpooled=np.zeros([total_no_of_regions],dtype=np.float32)
d1d2matches_regionallyweighted=np.zeros([total_no_of_regions],dtype=np.float32)

matched_local_pairs=[]

# Please modify according to your needs
dataset_name='Campus_loop'
save_visual_matches_dir='Visual_Matches/'+dataset_name+'/'

# Create directory for visual matches if it doesn't exist
if not os.path.exists(save_visual_matches_dir):
	os.makedirs(save_visual_matches_dir)    

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory = '/home/mihnea/datasets/campus_loop_original/live/'
ref_directory = '/home/mihnea/datasets/campus_loop_original/memory/'

# Please modify. This directory is for visualizing the entropy-based regions extraction
out_directory = '/media/mihnea/ConvSequential-SLAM/'


#For visualizing the correct and incorrect matches
def save_visual_matches(query,GT,retrieved):

    query_img=cv2.imread(query_directory+get_query_image_name(query))
    query_img=cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    gt_img=cv2.imread(ref_directory+get_ref_image_name(GT))
    gt_img=cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    retrieved_img=cv2.imread(ref_directory+get_ref_image_name(retrieved))
    retrieved_img=cv2.cvtColor(retrieved_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure()

    ax1 = fig.add_subplot(131)  
    plt.axis('off')
    ax1.imshow(query_img)
    ax2 = fig.add_subplot(132)
    plt.axis('off')
    ax2.imshow(retrieved_img)
    ax3 = fig.add_subplot(133)
    plt.axis('off')
    ax3.imshow(gt_img)
    ax1.title.set_text('Query Image')
    ax2.title.set_text('Retrieved Image')
    ax3.title.set_text('Ground-Truth')

#    plt.show()
    
    fig.savefig(save_visual_matches_dir+str(query)+'.jpg',bbox_inches='tight')
    plt.close(fig)

def largest_indices_thresholded(ary):
    good_list = np.where(ary>=ET)

    return good_list  

# Returns the n largest indices from a numpy array
def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]

    return np.unravel_index(indices, ary.shape)
       
def get_query_image_name(query):
    query_name = str(query + query_index_offset)
     
    return query_name + '.jpg'

def get_ref_image_name(ref):
    ref_name = str(ref + ref_index_offset)
    
    return ref_name + '.jpg'


def conv_match_dotproduct(d1,d2,regional_gd,total_no_of_regions): #Assumed aspect 1:1 here
     
    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs
    	
    np.dot(d1,d2, out=d1d2dot_matrix)

    #Select best matched ref region for every query region
    np.max(d1d2dot_matrix,axis=1,out=d1d2matches_maxpooled)  
             
    # Weighting regional matches with regional goodness
    np.multiply(d1d2matches_maxpooled,regional_gd,out=d1d2matches_regionallyweighted)   

    # Compute final match score
    score=np.sum(d1d2matches_regionallyweighted)/np.sum(regional_gd)    

    return score

# Function for matching a query-reference descriptor pair; returns the average score
def template_match(query_template, ref_template, regional_gd, total_no_of_regions):
	score = 0

	for ref in range(len(ref_template)):
		
		match_score = conv_match_dotproduct(query_template[ref],ref_template[ref],regional_gd,total_no_of_regions)
		score = score + match_score
		print(match_score)

	return score / k

# Load all reference images in ref_list
ref_list = []
for ref in range(total_Ref_Images):
    
    try:
        img_1 = cv2.imread(ref_directory+get_ref_image_name(ref), 0)

    except (IOError, ValueError) as e:
        img_1=None
        print('Exception! \n \n \n \n',ref)        
        
    if (img_1 is not None):
	
        img_1=cv2.resize(img_1,(magic_height,magic_width))
        startencodetimer=time.time()
        
        height,width,angle_unit=initialize(img_1, cell_size, bin_size)
	
        vector_1 = extract()
    
        vector_1=np.asfortranarray(vector_1.transpose(),dtype=np.float32)
        ref_list.append(vector_1)
print(len(ref_list))	

var = 0
ref_template_list = [[]]

while (var <= N_ref):
	k_number_list = []

	for i in range (var, var + k):
		k_number_list.append(ref_list[i])

	ref_template_list.append(k_number_list)
	var = var + 1	

ref_template_list.remove(ref_template_list[0])
 
# Load all query images in query_list
query_list = []
regional_goodness_list = []

for query in range(total_Query_Images):
    confusion_vector=np.zeros(total_Ref_Images,dtype=np.float32)

    try:    
        img_2 = cv2.imread(query_directory+get_query_image_name(query), 0)
        img_2rgb=cv2.imread(query_directory+get_query_image_name(query))
        
    except (IOError, ValueError) as e:
        img_2=None        
        print('Exception! \n \n \n \n')    
       
    if (img_2 is not None):
	
        img_2=cv2.resize(img_2,(magic_height,magic_width))
        img_2rgb=cv2.resize(img_2rgb,(magic_height,magic_width))
        
        startencodetimer=time.time()
 
        height,width,angle_unit=initialize(img_2, cell_size, bin_size)
	
        vector_2 = extract()
        vector_2=np.asfortranarray(vector_2,dtype=np.float32)
        query_list.append(vector_2)

        # Entropy Map
        img_gray = cv2.resize(img_as_ubyte(img_2),(100,100))
        ent_time=time.time()
        entropy_image=cv2.resize(entropy(img_gray, disk(5)),(magic_width,magic_height))
        
        # Finding Regions
        local_goodness=np.zeros([magic_height/cell_size-1,magic_width/cell_size-1],dtype=np.float32)
        
        for a in range (magic_height/cell_size-1):
            for b in range (magic_width/cell_size-1):
                local_staticity=1 # Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a*cell_size:a*cell_size + 2*cell_size, b*cell_size:b*cell_size + 2*cell_size])/(8*(cell_size*4*cell_size))

                if (local_entropy>=ET):    
                    local_goodness[a,b]=1
                else :
                    local_goodness[a,b]=0    
                    
        regional_goodness=local_goodness.flatten()
        regional_goodness_list.append(regional_goodness)
        regions = largest_indices_thresholded(local_goodness)
        no_of_good_regions=np.sum(regional_goodness)
        encodetime=time.time() - startencodetimer


var = 0
query_template_list = [[]]
while (var <= N_query):
	k_number_list = []	
	for i in range (var, var + k):
		k_number_list.append(query_list[i])
	query_template_list.append(k_number_list)
	var = var + 1

query_template_list.remove(query_template_list[0])

for query_itr in range(len(query_template_list)):
	ref_template_score = []
	for ref_itr in range(len(ref_template_list)):

		template_match_score = template_match(query_template_list[query_itr], ref_template_list[ref_itr], regional_goodness_list[query_itr], total_no_of_regions)
		ref_template_score.append(template_match_score)
        
	with open('Results_ConvSequential-SLAM_k_3.csv', 'a') as csvfile:
		my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		row= str(query_itr) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
		my_writer.writerow([row])

	# Save visualization 
    	save_visual_matches(
        	query_itr,                      # query
        	query_itr,                      # GT
        	np.argmax(ref_template_score)   # retrieved
    	)
