cur_id = -1
cur_num = 0
while line  = gets
  vals = line.split(",")
  if vals[0].to_i != cur_id
    if cur_id != -1
      print cur_id.to_s + "," + cur_num.to_s + "\n"
    end
    if vals[0] == "enrollment_id"
      next
    end
    cur_id = vals[0].to_i
    cur_num = 0
  end
  
  if vals[5].index("access")
    cur_num += 1
  end
end

# for last enroll id
print cur_id.to_s + "," + cur_num.to_s + "\n"

