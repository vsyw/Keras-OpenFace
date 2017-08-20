require 'csvigo'
require 'path'
require 'torch'
require 'nn'
require 'dpnn'
require 'image'
require 'paths'

function exportWeight (name, module, layer)
  if layer == 'SpatialConvolution' then 
    local w = module.weight
    local b = module.bias
    local shape = module.weight:size()
    local flatten = 1
    for i=1, shape:size() do
      flatten = flatten * shape[i]
    end
    w = torch.reshape(w, flatten)
    local w_csv = csvigo.File(paths.concat('./weights', name .. "_w.csv"), 'w')
    w_csv:write(w:totable())
    w_csv:close()
    local b_csv = csvigo.File(paths.concat('./weights', name .. "_b.csv"), 'w')
    b_csv:write(b:totable())
    b_csv:close()
    print(shape)
  end
  if layer == 'SpatialBatchNormalization' then
    local w = module.weight
    local b = module.bias
    local m = module.running_mean
    local v = module.running_var
    local w_csv = csvigo.File(paths.concat('./weights', name .. "_w.csv"), 'w')
    local b_csv = csvigo.File(paths.concat('./weights', name .. "_b.csv"), 'w')
    local m_csv = csvigo.File(paths.concat('./weights', name .. "_m.csv"), 'w')
    local v_csv = csvigo.File(paths.concat('./weights', name .. "_v.csv"), 'w')
    w_csv:write(w:totable())
    b_csv:write(b:totable())
    m_csv:write(m:totable())
    v_csv:write(v:totable())
    w_csv:close()
    b_csv:close()
    m_csv:close()
    v_csv:close()
  end
end
