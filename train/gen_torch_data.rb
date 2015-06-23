cur_id = -1
prob_num = 0
video_num = 0
access_num = 0
wiki_num = 0
discussion_num = 0
navigate_num = 0
page_close_num = 0
while line  = gets
  vals = line.split(",")
  if vals[0].to_i != cur_id
    if cur_id != -1
      all_num = prob_num + video_num + access_num + wiki_num + discussion_num + navigate_num + page_close_num
      print cur_id.to_s + "," + prob_num.to_s + "," + video_num.to_s + "," + access_num.to_s + "," + wiki_num.to_s + "," + discussion_num.to_s + "," + navigate_num.to_s + "," + page_close_num.to_s + "," + all_num.to_s + "\n"
    end
    if vals[0] == "enrollment_id"
      next
    end
    cur_id = vals[0].to_i
    prob_num = 0
    video_num = 0
    access_num = 0
    wiki_num = 0
    discussion_num = 0
    navigate_num = 0
    page_close_num = 0
  end

  if vals[5].index("problem")
    prob_num += 1
  end
  if vals[5].index("video")
    video_num += 1
  end    
  if vals[5].index("access")
    access_num += 1
  end
  if vals[5].index("wiki")
    wiki_num += 1
  end
  if vals[5].index("discussion")
    discussion_num += 1
  end
  if vals[5].index("navigate")
    navigate_num += 1
  end
  if vals[5].index("page_close")
    page_close_num += 1
  end
end

# for last enroll id
print cur_id.to_s + "," + prob_num.to_s + "," + video_num.to_s + "," + access_num.to_s + "," + wiki_num.to_s + "," + discussion_num.to_s + "," + navigate_num.to_s + "," + page_close_num.to_s + "," + all_num.to_s + "\n"

