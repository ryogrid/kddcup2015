require "time"

datas = Array.new
lec_users_hash = Hash.new

# enrollid_hash = Hash.new
# open("./truth_train.csv","r"){ |f|
#   while line  = f.gets
#    vals = line.split(",")
#    enrollid_hash[vals[0]] = 1
#   end
# }

# enrollid_time_hash = Hash.new
# vals = nil
# cur_id = -1
# cur_time = nil
# open("./log_train.csv","r"){ |f|
#   while line  = f.gets
#     vals = line.split(",")
#     if vals[0] != "enrollment_id" and cur_id != vals[0].to_i and cur_id != -1
#       enrollid_time_hash[vals[0]] = Time.strptime(cur_time, "%Y-%m-%dT%H:%M").to_i - 1401548400
#       cur_id = vals[0].to_i
#     end
#     cur_time = vals[3]
#   end
# }
# enrollid_time_hash[vals[0]] = Time.strptime(cur_time, "%Y-%m-%dT%H:%M").to_i - 1401548400

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
#  if enrollid_hash[datas[idx][0]] != nil
  print datas[idx][0] + "," + lec_users_hash[datas[idx][2]].to_s + "\n"
#  print datas[idx][0] + "," + lec_users_hash[datas[idx][2]].to_s + "," + enrollid_time_hash[datas[idx][0]].to_s + "\n"
#  end
}
