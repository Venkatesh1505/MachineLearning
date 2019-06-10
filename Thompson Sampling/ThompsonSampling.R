
#Read csv data
df = read.csv('Ads_CTR_Optimisation.csv')

#Thompson sampling
ads_selected = integer(0)
N = 10000
d = 10
number_of_random_1 = integer(d)
number_of_random_0 = integer(d)
total_reward = 0
for (n in 1:N){
  ad = 0
  max_random = 0
  for (i in 1:d){
    random_beta = rbeta(n = 1,
                        shape1 = number_of_random_1[i]+1,
                        shape2 = number_of_random_0[i]+1)
    if (random_beta > max_random){
      max_random = random_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = df[n, ad]
  if (reward == 1){
    number_of_random_1[ad] = number_of_random_1[ad] + 1
  }
  else{
    number_of_random_0[ad] = number_of_random_0[ad] + 1
  }
  total_reward = total_reward + reward
}

