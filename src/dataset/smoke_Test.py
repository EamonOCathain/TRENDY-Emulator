ds = CustomDataset(data_dir=..., std_dict=std_dict, tensor_type="train", exclude_vars=["foo","bar"])
print("input_dim:", len(ds.input_order), "output_dim:", len(ds.output_order))
print("schema sig:", ds.schema.signature())
x, m, a = ds[0]
print(x.shape, m.shape, a.shape)