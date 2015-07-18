cur_id = -1

event_str = ""
while line  = gets
  vals = line.split(",")
  if vals[0].to_i != cur_id
    if cur_id != -1
      print cur_id.to_s + event_str + "\n"
    end
    if vals[0] == "enrollment_id"
      next
    end
    cur_id = vals[0].to_i
    event_str = ""
  end

  if vals[5].index("problem")
    event_str = event_str + ",1"
  end
  if vals[5].index("video")
    event_str = event_str + ",2"
  end    
  if vals[5].index("access")
    event_str = event_str + ",3"
  end
  if vals[5].index("wiki")
    event_str = event_str + ",4"
  end
  if vals[5].index("discussion")
    event_str = event_str + ",5"
  end
  if vals[5].index("navigate")
    event_str = event_str + ",6"
  end
  if vals[5].index("page_close")
    event_str = event_str + ",7"
  end
end

# for last enroll id
print cur_id.to_s + event_str + "\n"

