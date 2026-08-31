from foundry_local_sdk import Configuration, FoundryLocalManager

config = Configuration(app_name="test")
FoundryLocalManager.initialize(config)
m = FoundryLocalManager.instance
catalog = m.catalog

all_models = catalog.list_models()
for model in all_models:
    if "phi" in model.alias.lower():
        print("ALIAS:", model.alias)
        for v in model.variants:
            print("  VARIANT:", v.id, "| cached:", v.is_cached)