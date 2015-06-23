datas = Array.new
lec_users_hash = Hash.new

enrollid_hash = Hash.new
open("./truth_train.csv","r"){ |f|
  while line  = f.gets
   vals = line.split(",")
   enrollid_hash[vals[0]] = 1
  end
}

while line  = gets
  vals = line.split(",")
  if vals[0] == "enrollment_id"
    next
  end
  
  datas << vals
  if lec_users_hash[vals[2]] == nil
    lec_users_hash[vals[2]] = 1
  else
    lec_users_hash[vals[2]] = lec_users_hash[vals[2]] + 1
  end
end

datas.each_index{ |idx|
  if enrollid_hash[datas[idx][0]] != nil
    print datas[idx][0] + "," + lec_users_hash[datas[idx][2]].to_s + "\n"
  end
}
