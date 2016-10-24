import os 
import os.path
import shutil
import glob

#------make directory every 5 degree----------------------------------

#i = 1
#while i<=100:
#	#os.mkdir("./obj" + str(i))
#	os.mkdir("./obj" + str(i) + "/train_" + str(i))
#	os.mkdir("./obj" + str(i) + "/test_" + str(i))
#	i += 1





#------classify objects-----------------------------------------------

filename = glob.glob('/home/tomoya/coil-100/all_objects/*.png')
#print filename

#for name in filename:
	#print os.path.basename(name)

for name in filename:
	print name 
        path = os.path.basename(name)
	print "path = " + path
	num_underbar = path.find("_")
	print "position'_' = " +  str(num_underbar)
	 
	print  path[:num_underbar]
	object_name = path[:num_underbar]
	object_num = object_name[3:]
	print object_num
	
	num_dot = path.find(".")
	print "position'.' = " + str(num_dot)
	print path[num_underbar+2:num_dot]
	degree = path[num_underbar+2:num_dot]

	if int(degree)+40 <= 320 and (int(degree) + 40) % 40 == 0:  
		shutil.copy(name, "/home/tomoya/coil-100/"+object_name+"/test_" + object_num)
	else:
		shutil.copy(name, "/home/tomoya/coil-100/"+object_name+"/train_" + object_num)
	
	
		




