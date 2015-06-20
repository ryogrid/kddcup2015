while line  = gets
  vals = line.split(",")
  prob = 1 - (vals[1].to_i / 500.0)
  if prob < 0
    prob = 0
  end
  print vals[0] + "," + prob.to_s + "\n"
end
