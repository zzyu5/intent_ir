import triton.language as tl


class _TLExt:
    def program_id(self, dim):
        return tl.program_id(dim)


triton_lang_extension = _TLExt()
# Expose raw tl.program_id to avoid runtime errors
program_id = tl.program_id

__all__ = ["triton_lang_extension", "program_id"]
