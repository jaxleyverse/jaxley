from morph_tool import convert


def run_conversion():
    convert("morph_tmp.swc", "morph.swc", sanitize=True, single_point_soma=True)


if __name__ == "__main__":
    run_conversion()
