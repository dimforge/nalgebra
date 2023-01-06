#![allow(missing_docs)]

//! Traits and tags for identifying the dimension of all algebraic entities.

use std::any::{Any, TypeId};
use std::cmp;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use typenum::{self, Diff, Max, Maximum, Min, Minimum, Prod, Quot, Sum, Unsigned};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Dim of dynamically-sized algebraic entities.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-serialize",
    archive_attr(derive(bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
pub struct Dynamic {
    value: usize,
}

impl Dynamic {
    /// A dynamic size equal to `value`.
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self { value }
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl Serialize for Dynamic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'de> Deserialize<'de> for Dynamic {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        usize::deserialize(deserializer).map(|x| Dynamic { value: x })
    }
}

/// Trait implemented by `Dynamic`.
pub trait IsDynamic {}
/// Trait implemented by `Dynamic` and type-level integers different from `U1`.
pub trait IsNotStaticOne {}

impl IsDynamic for Dynamic {}
impl IsNotStaticOne for Dynamic {}

/// Trait implemented by any type that can be used as a dimension. This includes type-level
/// integers and `Dynamic` (for dimensions not known at compile-time).
pub unsafe trait Dim: Any + Debug + Copy + PartialEq + Send + Sync {
    #[inline(always)]
    fn is<D: Dim>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<D>()
    }

    /// Gets the compile-time value of `Self`. Returns `None` if it is not known, i.e., if `Self =
    /// Dynamic`.
    fn try_to_usize() -> Option<usize>;

    /// Gets the run-time value of `self`. For type-level integers, this is the same as
    /// `Self::try_to_usize().unwrap()`.
    fn value(&self) -> usize;

    /// Builds an instance of `Self` from a run-time value. Panics if `Self` is a type-level
    /// integer and `dim != Self::try_to_usize().unwrap()`.
    fn from_usize(dim: usize) -> Self;
}

unsafe impl Dim for Dynamic {
    #[inline]
    fn try_to_usize() -> Option<usize> {
        None
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        Self::new(dim)
    }

    #[inline]
    fn value(&self) -> usize {
        self.value
    }
}

impl Add<usize> for Dynamic {
    type Output = Dynamic;

    #[inline]
    fn add(self, rhs: usize) -> Self {
        Self::new(self.value + rhs)
    }
}

impl Sub<usize> for Dynamic {
    type Output = Dynamic;

    #[inline]
    fn sub(self, rhs: usize) -> Self {
        Self::new(self.value - rhs)
    }
}

/*
 *
 * Operations.
 *
 */

macro_rules! dim_ops(
    ($($DimOp:    ident, $DimNameOp: ident,
       $Op:       ident, $op: ident, $op_path: path,
       $DimResOp: ident, $DimNameResOp: ident,
       $ResOp: ident);* $(;)*) => {$(
        pub type $DimResOp<D1, D2> = <D1 as $DimOp<D2>>::Output;

        pub trait $DimOp<D: Dim>: Dim {
            type Output: Dim;

            fn $op(self, other: D) -> Self::Output;
        }

        impl<const A: usize, const B: usize> $DimOp<Const<B>> for Const<A>
        where
            Const<A>: ToTypenum,
            Const<B>: ToTypenum,
            <Const<A> as ToTypenum>::Typenum: $Op<<Const<B> as ToTypenum>::Typenum>,
            $ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum>: ToConst,
        {
            type Output =
                <$ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum> as ToConst>::Const;

            fn $op(self, _: Const<B>) -> Self::Output {
                Self::Output::name()
            }
        }

        impl<D: Dim> $DimOp<D> for Dynamic {
            type Output = Dynamic;

            #[inline]
            fn $op(self, other: D) -> Dynamic {
                Dynamic::new($op_path(self.value, other.value()))
            }
        }

        // TODO: use Const<T> instead of D: DimName?
        impl<D: DimName> $DimOp<Dynamic> for D {
            type Output = Dynamic;

            #[inline]
            fn $op(self, other: Dynamic) -> Dynamic {
                Dynamic::new($op_path(self.value(), other.value))
            }
        }

        pub type $DimNameResOp<D1, D2> = <D1 as $DimNameOp<D2>>::Output;

        pub trait $DimNameOp<D: DimName>: DimName {
            type Output: DimName;

            fn $op(self, other: D) -> Self::Output;
        }

        impl<const A: usize, const B: usize> $DimNameOp<Const<B>> for Const<A>
        where
            Const<A>: ToTypenum,
            Const<B>: ToTypenum,
            <Const<A> as ToTypenum>::Typenum: $Op<<Const<B> as ToTypenum>::Typenum>,
            $ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum>: ToConst,
        {
            type Output =
                <$ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum> as ToConst>::Const;

            fn $op(self, _: Const<B>) -> Self::Output {
                Self::Output::name()
            }
        }
   )*}
);

dim_ops!(
    DimAdd, DimNameAdd, Add, add, Add::add, DimSum,     DimNameSum,     Sum;
    DimMul, DimNameMul, Mul, mul, Mul::mul, DimProd,    DimNameProd,    Prod;
    DimSub, DimNameSub, Sub, sub, Sub::sub, DimDiff,    DimNameDiff,    Diff;
    DimDiv, DimNameDiv, Div, div, Div::div, DimQuot,    DimNameQuot,    Quot;
    DimMin, DimNameMin, Min, min, cmp::min, DimMinimum, DimNameMinimum, Minimum;
    DimMax, DimNameMax, Max, max, cmp::max, DimMaximum, DimNameMaximum, Maximum;
);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "rkyv-serialize-no-std",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-serialize",
    archive_attr(derive(bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
pub struct Const<const R: usize>;

/// Trait implemented exclusively by type-level integers.
pub trait DimName: Dim {
    const USIZE: usize;

    /// The name of this dimension, i.e., the singleton `Self`.
    fn name() -> Self;

    // TODO: this is not a very idiomatic name.
    /// The value of this dimension.
    fn dim() -> usize;
}

#[cfg(feature = "serde-serialize-no-std")]
impl<const D: usize> Serialize for Const<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ().serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'de, const D: usize> Deserialize<'de> for Const<D> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'de>,
    {
        <()>::deserialize(deserializer).map(|_| Const::<D>)
    }
}

pub trait ToConst {
    type Const: DimName;
}

pub trait ToTypenum {
    type Typenum: Unsigned;
}

unsafe impl<const T: usize> Dim for Const<T> {
    #[inline]
    fn try_to_usize() -> Option<usize> {
        Some(T)
    }

    #[inline]
    fn value(&self) -> usize {
        T
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        assert_eq!(dim, T);
        Self
    }
}

impl<const T: usize> DimName for Const<T> {
    const USIZE: usize = T;

    #[inline]
    fn name() -> Self {
        Self
    }

    #[inline]
    fn dim() -> usize {
        T
    }
}

pub type U1 = Const<1>;

impl ToTypenum for Const<1> {
    type Typenum = typenum::U1;
}

impl ToConst for typenum::U1 {
    type Const = Const<1>;
}

macro_rules! from_to_typenum (
    ($($D: ident, $VAL: expr);* $(;)*) => {$(
        pub type $D = Const<$VAL>;

        impl ToTypenum for Const<$VAL> {
            type Typenum = typenum::$D;
        }

        impl ToConst for typenum::$D {
            type Const = Const<$VAL>;
        }

        impl IsNotStaticOne for $D { }
    )*}
);

from_to_typenum!(
    U0, 0; /*U1,1;*/ U2, 2; U3, 3; U4, 4; U5, 5; U6, 6; U7, 7; U8, 8; U9, 9; U10, 10; U11, 11; U12, 12; U13, 13; U14, 14; U15, 15; U16, 16; U17, 17; U18, 18;
    U19, 19; U20, 20; U21, 21; U22, 22; U23, 23; U24, 24; U25, 25; U26, 26; U27, 27; U28, 28; U29, 29; U30, 30; U31, 31; U32, 32; U33, 33; U34, 34; U35, 35; U36, 36; U37, 37;
    U38, 38; U39, 39; U40, 40; U41, 41; U42, 42; U43, 43; U44, 44; U45, 45; U46, 46; U47, 47; U48, 48; U49, 49; U50, 50; U51, 51; U52, 52; U53, 53; U54, 54; U55, 55; U56, 56;
    U57, 57; U58, 58; U59, 59; U60, 60; U61, 61; U62, 62; U63, 63; U64, 64; U65, 65; U66, 66; U67, 67; U68, 68; U69, 69; U70, 70; U71, 71; U72, 72; U73, 73; U74, 74; U75, 75;
    U76, 76; U77, 77; U78, 78; U79, 79; U80, 80; U81, 81; U82, 82; U83, 83; U84, 84; U85, 85; U86, 86; U87, 87; U88, 88; U89, 89; U90, 90; U91, 91; U92, 92; U93, 93; U94, 94;
    U95, 95; U96, 96; U97, 97; U98, 98; U99, 99; U100, 100; U101, 101; U102, 102; U103, 103; U104, 104; U105, 105; U106, 106; U107, 107; U108, 108; U109, 109; U110, 110;
    U111, 111; U112, 112; U113, 113; U114, 114; U115, 115; U116, 116; U117, 117; U118, 118; U119, 119; U120, 120; U121, 121; U122, 122; U123, 123; U124, 124; U125, 125; U126, 126;
    U127, 127;
);

#[cfg(feature = "const-generics-1024")]
from_to_typenum!(
    U128, 128; U129, 129; U130, 130; U131, 131; U132, 132; U133, 133; U134, 134; U135, 135; U136, 136; U137, 137; U138, 138; U139, 139; U140, 140; U141, 141; U142, 142; 
    U143, 143; U144, 144; U145, 145; U146, 146; U147, 147; U148, 148; U149, 149; U150, 150; U151, 151; U152, 152; U153, 153; U154, 154; U155, 155; U156, 156; U157, 157; U158, 158;
    U159, 159; U160, 160; U161, 161; U162, 162; U163, 163; U164, 164; U165, 165; U166, 166; U167, 167; U168, 168; U169, 169; U170, 170; U171, 171; U172, 172; U173, 173; U174, 174; 
    U175, 175; U176, 176; U177, 177; U178, 178; U179, 179; U180, 180; U181, 181; U182, 182; U183, 183; U184, 184; U185, 185; U186, 186; U187, 187; U188, 188; U189, 189; U190, 190; 
    U191, 191; U192, 192; U193, 193; U194, 194; U195, 195; U196, 196; U197, 197; U198, 198; U199, 199; U200, 200; U201, 201; U202, 202; U203, 203; U204, 204; U205, 205; U206, 206;
    U207, 207; U208, 208; U209, 209; U210, 210; U211, 211; U212, 212; U213, 213; U214, 214; U215, 215; U216, 216; U217, 217; U218, 218; U219, 219; U220, 220; U221, 221; U222, 222;
    U223, 223; U224, 224; U225, 225; U226, 226; U227, 227; U228, 228; U229, 229; U230, 230; U231, 231; U232, 232; U233, 233; U234, 234; U235, 235; U236, 236; U237, 237; U238, 238; 
    U239, 239; U240, 240; U241, 241; U242, 242; U243, 243; U244, 244; U245, 245; U246, 246; U247, 247; U248, 248; U249, 249; U250, 250; U251, 251; U252, 252; U253, 253; U254, 254; 
    U255, 255; U256, 256; U257, 257; U258, 258; U259, 259; U260, 260; U261, 261; U262, 262; U263, 263; U264, 264; U265, 265; U266, 266; U267, 267; U268, 268; U269, 269; U270, 270; 
    U271, 271; U272, 272; U273, 273; U274, 274; U275, 275; U276, 276; U277, 277; U278, 278; U279, 279; U280, 280; U281, 281; U282, 282; U283, 283; U284, 284; U285, 285; U286, 286;
    U287, 287; U288, 288; U289, 289; U290, 290; U291, 291; U292, 292; U293, 293; U294, 294; U295, 295; U296, 296; U297, 297; U298, 298; U299, 299; U300, 300; U301, 301; U302, 302;
    U303, 303; U304, 304; U305, 305; U306, 306; U307, 307; U308, 308; U309, 309; U310, 310; U311, 311; U312, 312; U313, 313; U314, 314; U315, 315; U316, 316; U317, 317; U318, 318;
    U319, 319; U320, 320; U321, 321; U322, 322; U323, 323; U324, 324; U325, 325; U326, 326; U327, 327; U328, 328; U329, 329; U330, 330; U331, 331; U332, 332; U333, 333; U334, 334;
    U335, 335; U336, 336; U337, 337; U338, 338; U339, 339; U340, 340; U341, 341; U342, 342; U343, 343; U344, 344; U345, 345; U346, 346; U347, 347; U348, 348; U349, 349; U350, 350;
    U351, 351; U352, 352; U353, 353; U354, 354; U355, 355; U356, 356; U357, 357; U358, 358; U359, 359; U360, 360; U361, 361; U362, 362; U363, 363; U364, 364; U365, 365; U366, 366; 
    U367, 367; U368, 368; U369, 369; U370, 370; U371, 371; U372, 372; U373, 373; U374, 374; U375, 375; U376, 376; U377, 377; U378, 378; U379, 379; U380, 380; U381, 381; U382, 382;
    U383, 383; U384, 384; U385, 385; U386, 386; U387, 387; U388, 388; U389, 389; U390, 390; U391, 391; U392, 392; U393, 393; U394, 394; U395, 395; U396, 396; U397, 397; U398, 398;
    U399, 399; U400, 400; U401, 401; U402, 402; U403, 403; U404, 404; U405, 405; U406, 406; U407, 407; U408, 408; U409, 409; U410, 410; U411, 411; U412, 412; U413, 413; U414, 414;
    U415, 415; U416, 416; U417, 417; U418, 418; U419, 419; U420, 420; U421, 421; U422, 422; U423, 423; U424, 424; U425, 425; U426, 426; U427, 427; U428, 428; U429, 429; U430, 430;
    U431, 431; U432, 432; U433, 433; U434, 434; U435, 435; U436, 436; U437, 437; U438, 438; U439, 439; U440, 440; U441, 441; U442, 442; U443, 443; U444, 444; U445, 445; U446, 446;
    U447, 447; U448, 448; U449, 449; U450, 450; U451, 451; U452, 452; U453, 453; U454, 454; U455, 455; U456, 456; U457, 457; U458, 458; U459, 459; U460, 460; U461, 461; U462, 462;
    U463, 463; U464, 464; U465, 465; U466, 466; U467, 467; U468, 468; U469, 469; U470, 470; U471, 471; U472, 472; U473, 473; U474, 474; U475, 475; U476, 476; U477, 477; U478, 478;
    U479, 479; U480, 480; U481, 481; U482, 482; U483, 483; U484, 484; U485, 485; U486, 486; U487, 487; U488, 488; U489, 489; U490, 490; U491, 491; U492, 492; U493, 493; U494, 494;
    U495, 495; U496, 496; U497, 497; U498, 498; U499, 499; U500, 500; U501, 501; U502, 502; U503, 503; U504, 504; U505, 505; U506, 506; U507, 507; U508, 508; U509, 509; U510, 510;
    U511, 511; U512, 512; U513, 513; U514, 514; U515, 515; U516, 516; U517, 517; U518, 518; U519, 519; U520, 520; U521, 521; U522, 522; U523, 523; U524, 524; U525, 525; U526, 526;
    U527, 527; U528, 528; U529, 529; U530, 530; U531, 531; U532, 532; U533, 533; U534, 534; U535, 535; U536, 536; U537, 537; U538, 538; U539, 539; U540, 540; U541, 541; U542, 542;
    U543, 543; U544, 544; U545, 545; U546, 546; U547, 547; U548, 548; U549, 549; U550, 550; U551, 551; U552, 552; U553, 553; U554, 554; U555, 555; U556, 556; U557, 557; U558, 558;
    U559, 559; U560, 560; U561, 561; U562, 562; U563, 563; U564, 564; U565, 565; U566, 566; U567, 567; U568, 568; U569, 569; U570, 570; U571, 571; U572, 572; U573, 573; U574, 574;
    U575, 575; U576, 576; U577, 577; U578, 578; U579, 579; U580, 580; U581, 581; U582, 582; U583, 583; U584, 584; U585, 585; U586, 586; U587, 587; U588, 588; U589, 589; U590, 590;
    U591, 591; U592, 592; U593, 593; U594, 594; U595, 595; U596, 596; U597, 597; U598, 598; U599, 599; U600, 600; U601, 601; U602, 602; U603, 603; U604, 604; U605, 605; U606, 606;
    U607, 607; U608, 608; U609, 609; U610, 610; U611, 611; U612, 612; U613, 613; U614, 614; U615, 615; U616, 616; U617, 617; U618, 618; U619, 619; U620, 620; U621, 621; U622, 622;
    U623, 623; U624, 624; U625, 625; U626, 626; U627, 627; U628, 628; U629, 629; U630, 630; U631, 631; U632, 632; U633, 633; U634, 634; U635, 635; U636, 636; U637, 637; U638, 638;
    U639, 639; U640, 640; U641, 641; U642, 642; U643, 643; U644, 644; U645, 645; U646, 646; U647, 647; U648, 648; U649, 649; U650, 650; U651, 651; U652, 652; U653, 653; U654, 654;
    U655, 655; U656, 656; U657, 657; U658, 658; U659, 659; U660, 660; U661, 661; U662, 662; U663, 663; U664, 664; U665, 665; U666, 666; U667, 667; U668, 668; U669, 669; U670, 670;
    U671, 671; U672, 672; U673, 673; U674, 674; U675, 675; U676, 676; U677, 677; U678, 678; U679, 679; U680, 680; U681, 681; U682, 682; U683, 683; U684, 684; U685, 685; U686, 686;
    U687, 687; U688, 688; U689, 689; U690, 690; U691, 691; U692, 692; U693, 693; U694, 694; U695, 695; U696, 696; U697, 697; U698, 698; U699, 699; U700, 700; U701, 701; U702, 702;
    U703, 703; U704, 704; U705, 705; U706, 706; U707, 707; U708, 708; U709, 709; U710, 710; U711, 711; U712, 712; U713, 713; U714, 714; U715, 715; U716, 716; U717, 717; U718, 718;
    U719, 719; U720, 720; U721, 721; U722, 722; U723, 723; U724, 724; U725, 725; U726, 726; U727, 727; U728, 728; U729, 729; U730, 730; U731, 731; U732, 732; U733, 733; U734, 734;
    U735, 735; U736, 736; U737, 737; U738, 738; U739, 739; U740, 740; U741, 741; U742, 742; U743, 743; U744, 744; U745, 745; U746, 746; U747, 747; U748, 748; U749, 749; U750, 750;
    U751, 751; U752, 752; U753, 753; U754, 754; U755, 755; U756, 756; U757, 757; U758, 758; U759, 759; U760, 760; U761, 761; U762, 762; U763, 763; U764, 764; U765, 765; U766, 766;
    U767, 767; U768, 768; U769, 769; U770, 770; U771, 771; U772, 772; U773, 773; U774, 774; U775, 775; U776, 776; U777, 777; U778, 778; U779, 779; U780, 780; U781, 781; U782, 782;
    U783, 783; U784, 784; U785, 785; U786, 786; U787, 787; U788, 788; U789, 789; U790, 790; U791, 791; U792, 792; U793, 793; U794, 794; U795, 795; U796, 796; U797, 797; U798, 798;
    U799, 799; U800, 800; U801, 801; U802, 802; U803, 803; U804, 804; U805, 805; U806, 806; U807, 807; U808, 808; U809, 809; U810, 810; U811, 811; U812, 812; U813, 813; U814, 814;
    U815, 815; U816, 816; U817, 817; U818, 818; U819, 819; U820, 820; U821, 821; U822, 822; U823, 823; U824, 824; U825, 825; U826, 826; U827, 827; U828, 828; U829, 829; U830, 830;
    U831, 831; U832, 832; U833, 833; U834, 834; U835, 835; U836, 836; U837, 837; U838, 838; U839, 839; U840, 840; U841, 841; U842, 842; U843, 843; U844, 844; U845, 845; U846, 846;
    U847, 847; U848, 848; U849, 849; U850, 850; U851, 851; U852, 852; U853, 853; U854, 854; U855, 855; U856, 856; U857, 857; U858, 858; U859, 859; U860, 860; U861, 861; U862, 862;
    U863, 863; U864, 864; U865, 865; U866, 866; U867, 867; U868, 868; U869, 869; U870, 870; U871, 871; U872, 872; U873, 873; U874, 874; U875, 875; U876, 876; U877, 877; U878, 878;
    U879, 879; U880, 880; U881, 881; U882, 882; U883, 883; U884, 884; U885, 885; U886, 886; U887, 887; U888, 888; U889, 889; U890, 890; U891, 891; U892, 892; U893, 893; U894, 894;
    U895, 895; U896, 896; U897, 897; U898, 898; U899, 899; U900, 900; U901, 901; U902, 902; U903, 903; U904, 904; U905, 905; U906, 906; U907, 907; U908, 908; U909, 909; U910, 910;
    U911, 911; U912, 912; U913, 913; U914, 914; U915, 915; U916, 916; U917, 917; U918, 918; U919, 919; U920, 920; U921, 921; U922, 922; U923, 923; U924, 924; U925, 925; U926, 926;
    U927, 927; U928, 928; U929, 929; U930, 930; U931, 931; U932, 932; U933, 933; U934, 934; U935, 935; U936, 936; U937, 937; U938, 938; U939, 939; U940, 940; U941, 941; U942, 942;
    U943, 943; U944, 944; U945, 945; U946, 946; U947, 947; U948, 948; U949, 949; U950, 950; U951, 951; U952, 952; U953, 953; U954, 954; U955, 955; U956, 956; U957, 957; U958, 958;
    U959, 959; U960, 960; U961, 961; U962, 962; U963, 963; U964, 964; U965, 965; U966, 966; U967, 967; U968, 968; U969, 969; U970, 970; U971, 971; U972, 972; U973, 973; U974, 974;
    U975, 975; U976, 976; U977, 977; U978, 978; U979, 979; U980, 980; U981, 981; U982, 982; U983, 983; U984, 984; U985, 985; U986, 986; U987, 987; U988, 988; U989, 989; U990, 990;
    U991, 991; U992, 992; U993, 993; U994, 994; U995, 995; U996, 996; U997, 997; U998, 998; U999, 999; U1000, 1000; U1001, 1001; U1002, 1002; U1003, 1003; U1004, 1004; U1005, 1005;
    U1006, 1006; U1007, 1007; U1008, 1008; U1009, 1009; U1010, 1010; U1011, 1011; U1012, 1012; U1013, 1013; U1014, 1014; U1015, 1015; U1016, 1016; U1017, 1017; U1018, 1018; 
    U1019, 1019; U1020, 1020; U1021, 1021; U1022, 1022; U1023, 1023; U1024, 1024
);


#[cfg(test)]
mod tests {
    use crate::{Const, ToTypenum};

    fn use_totypenum<T: ToTypenum>(_c: T) { }

    // By default, Const<R> implements ToTypenum conversion, where R in [0, 127]
    #[test]
    fn can_use_typenum_upto_127() {
        let c: Const<0> = Const{};
        use_totypenum(c);
        let c: Const<1> = Const{};
        use_totypenum(c);
        let c: Const<127> = Const{};
        use_totypenum(c);
    }

    // Behind a feature flag, Const<R> implements ToTypenum conversion, where R in [0, 1024]
    #[cfg(feature = "const-generics-1024")]
    #[test]
    fn can_use_typenum_upto_1024() {
        let c: Const<0> = Const{};
        use_totypenum(c);
        let c: Const<1> = Const{};
        use_totypenum(c);
        let c: Const<127> = Const{};
        use_totypenum(c);
        let c: Const<128> = Const{};
        use_totypenum(c);
        let c: Const<256> = Const{};
        use_totypenum(c);
        let c: Const<512> = Const{};
        use_totypenum(c);
        let c: Const<1023> = Const{};
        use_totypenum(c);
        let c: Const<1024> = Const{};
        use_totypenum(c);
    }
}