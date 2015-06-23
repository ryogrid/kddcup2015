datas = Array.new
user_lecs_hash = Hash.new

# enrollid_hash = Hash.new
# open("../test/truth_test.csv","r"){ |f|
#   while line  = f.gets
#     vals = line.split(",")
#     enrollid_hash[vals[0]] = 1
#   end
# }

while line  = gets
  vals = line.split(",")
  if vals[0] == "enrollment_id"
    next
  end
  
  datas << vals
  if user_lecs_hash[vals[1]] == nil
    user_lecs_hash[vals[1]] = 1
  else
    user_lecs_hash[vals[1]] = user_lecs_hash[vals[1]] + 1
  end
end

datas.each_index{ |idx|
#  if enrollid_hash[datas[idx][0]] != nil
    print datas[idx][0] + "," + user_lecs_hash[datas[idx][1]].to_s + "\n"
#  end
}
