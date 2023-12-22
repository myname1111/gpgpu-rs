#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntryType {
    Buffer,
    Uniform,
    ConstImage,
    Image,
}
