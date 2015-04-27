local manifold = require 'manifold'

local function mani()
  local N = 3200

  local mnist = require 'mnist'
  local train = mnist.traindataset()
  train.size = N

  train.data = train.data:narrow(1, 1, N)
  train.label = train.label:narrow(1, 1, N)
  local x = torch.Tensor(train.data:size())
  
  x:map(train.data, function(xx, yy) return yy end)
  x:resize(x:size(1), x:size(2) * x:size(3))
  local labels = train.label

  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = false}
  mapped_x1 = manifold.embedding.tsne(x, opts)
  print('t-SNE: ' .. timer:time().real .. ' seconds.')

  for i = 1,3200 do
    print((labels[i] + 1) .. ' ' .. mapped_x1[i][1] .. ' ' .. mapped_x1[i][2])
  end
end

mani()
