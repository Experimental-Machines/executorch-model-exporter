from pipeline import exporting


def meminfo(path, swap_free_kb, available_kb):
    path.write_text(
        f"MemTotal:       16373452 kB\nMemAvailable:   {available_kb} kB\n"
        f"SwapTotal:      25165820 kB\nSwapFree:       {swap_free_kb} kB\n"
    )


def test_memory_sampler_keeps_the_peak_swap_and_the_lowest_available_memory(tmp_path):
    path = tmp_path / "meminfo"
    meminfo(path, swap_free_kb=25165820, available_kb=12000000)
    sampler = exporting.MemorySampler(interval=3600, meminfo=path)
    sampler.sample()
    meminfo(path, swap_free_kb=20165820, available_kb=300000)  # 5,000,000 kB of swap in use
    sampler.sample()
    meminfo(path, swap_free_kb=25165820, available_kb=9000000)
    sampler.sample()
    assert sampler.result() == {
        "peak_swap_used_bytes": 5_000_000 * 1024,
        "min_mem_available_bytes": 300_000 * 1024,
    }


def test_memory_sampler_is_a_no_op_without_proc(tmp_path):
    with exporting.MemorySampler(interval=3600, meminfo=tmp_path / "missing") as sampler:
        pass
    assert sampler.result() == {"peak_swap_used_bytes": None, "min_mem_available_bytes": None}


def test_contains_is_shared_by_the_npu_backends(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"x" * 100 + b"NeuropilotBackend" + b"y" * 100)
    assert exporting.contains(path, b"NeuropilotBackend", block=7)
